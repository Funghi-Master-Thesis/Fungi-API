import datetime
from multiprocessing import freeze_support
import os
import time
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import cv2

import glob
from tqdm import tqdm

train_folder = r'.\Data\DataSet\training'
test_folder = r'.\Data\DataSet\test'

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3
BATCH_SIZE = 32
NUM_WORKERS = 2


MODEL_SAVE_LOC = r'..\data\04_model'
REPORT_SAVE_LOC = r'..\data\06_reporting'

class FungiDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, csv_file, root_dir="", transform=None):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        image_path = self.annotation_df.iloc[idx, 1] #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
        class_name = self.annotation_df.iloc[idx, 2] # use class name column (index = 2) in csv file
        class_index = self.annotation_df.iloc[idx, 3] # use class index column (index = 3) in csv file
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index
    # def visualize(self, number_of_img=10, output_width=12, output_height=6):
    #     plt.figure(figsize=(output_width, output_height))
    #     for i in range(number_of_img):
    #         idx = random.randint(0, len(self.annotation_df))
    #         image, class_name, class_index = self.__getitem__(idx)
    #         ax = plt.subplot(2, 5, i+1)  # create an axis
    #         # create a name of the axis based on the img name
    #         ax.title.set_text(class_name + '-' + str(class_index))
    #         if self.transform == None:
    #             plt.imshow(image)
    #         else:
    #             plt.imshow(image.permute(1, 2, 0))
    #     plt.show()

def create_validation_dataset(dataset, validation_proportion):
    if (validation_proportion > 1) or (validation_proportion < 0):
        return "The proportion of the validation set must be between 0 and 1"
    else:
        dataset_size = int((1 - validation_proportion) * len(dataset))
        validation_size = len(dataset) - dataset_size
        print(dataset_size, validation_size)
        dataset, validation_set = torch.utils.data.random_split(
            dataset, [dataset_size, validation_size])
        return dataset, validation_set

def default_loss():
    return nn.CrossEntropyLoss()

def default_optimizer(model, learning_rate = 0.001):
    return optim.Adam(model.parameters(), lr = learning_rate)

def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def model_prep_and_summary(model, device):
    """
    Move model to GPU and print model summary
    """
    # Define the model and move it to GPU:
    model = model
    model = model.to(device)
    print('Current device: ' + str(device))
    print('Is Model on CUDA: ' + str(next(model.parameters()).is_cuda))
    # Display model summary:
    summary(model, (INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))
                
    
class MyCnnModel(nn.Module):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Define a max pooling layer to use repeatedly in the forward function
        # The role of pooling layer is to reduce the spatial dimension (H, W) of the input volume for next layers.
        # It only affects weight and height but not depth.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # output shape of maxpool3 is 64*28*28
        self.fc14 = nn.Linear(64*28*28, 500)
        self.fc15 = nn.Linear(500, 50)
        # output of the final DC layer = 6 = number of classes
        self.fc16 = nn.Linear(50, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # maxpool1 output shape is 16*112*112 (112 = (224-2)/2 + 1)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # maxpool2 output shape is 32*56*56 (56 = (112-2)/2 + 1)
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        # maxpool3 output shape is 64*28*28 (28 = (56-2)/2 + 1)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x   
    
def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    # device = get_default_device()
    model = model.to(device)
    train_result_dict = {'epoch': [], 'train_loss': [],
                         'val_loss': [], 'accuracy': [], 'time': []}

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        model.train()  # set the model to training mode, parameters are updated
        for data in train_loader:
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(image)  # forward propagation
            loss = criterion(outputs, class_index)  # loss calculation
            loss.backward()  # backward propagation
            optimizer.step()  # params update
            train_loss += loss.item()  # loss for each minibatch
            _, predicted = torch.max(outputs.data, 1)
            total += class_index.size(0)
            correct += (predicted == class_index).sum().item()
            epoch_accuracy = round(float(correct)/float(total)*100, 2)

        # Here evaluation is combined together with
        val_loss = 0.0
        model.eval()  # set the model to evaluation mode, parameters are frozen
        for data in val_loader:
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            loss = criterion(outputs, class_index)
            val_loss += loss.item()

        # print statistics every 1 epoch
        # divide by the length of the minibatch because loss.item() returns the loss of the whole minibatch
        train_loss_result = round(train_loss / len(train_loader), 3)
        val_loss_result = round(val_loss / len(val_loader), 3)

        epoch_time = round(time.time() - start_time, 1)
        # add statistics to the dictionary:
        train_result_dict['epoch'].append(epoch + 1)
        train_result_dict['train_loss'].append(train_loss_result)
        train_result_dict['val_loss'].append(val_loss_result)
        train_result_dict['accuracy'].append(epoch_accuracy)
        train_result_dict['time'].append(epoch_time)

        print(f'Epoch {epoch+1} \t Training Loss: {train_loss_result} \t Validation Loss: {val_loss_result} \t Epoch Train Accuracy (%): {epoch_accuracy} \t Epoch Time (s): {epoch_time}')
    # return the trained model and the loss dictionary
    return model, train_result_dict


def visualize_training(train_result_dictionary):
    # Define Data
    df = pd.DataFrame(train_result_dictionary)
    x = df['epoch']
    data_1 = df['train_loss']
    data_2 = df['val_loss']
    data_3 = df['accuracy']

    # Create Plot
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(x, data_1, color='red', label='training loss')
    ax1.plot(x, data_2, color='blue', label='validation loss')

    # Adding Twin Axes
    ax2 = ax1.twinx()
    ax2.plot(x, data_3, color='green', label='Training Accuracy')

    # Add label
    plt.ylabel('Accuracy')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center')

    # Show plot
    plt.show() 
    
def infer(model, device, data_loader):
    '''
    Calculate predicted class indices of the data_loader by the trained model 
    '''
    model = model.to(device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in data_loader:
            image, class_name, class_index = data
            image = image.to(device)
            class_index = class_index.to(device)
            outputs = model(image)
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)
            class_index = class_index.data.cpu().numpy()
            y_true.extend(class_index)
    return y_pred, y_true

def infer_single_image(model, device, image_path, transform):
    '''
    Calculate predicted class index of the image by the trained model 
    '''
    # Prepare the Image
    image = cv2.imread(image_path)  # read image by cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform(image)
    plt.imshow(image_transformed.permute(1, 2, 0))
    image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)

    # Inference
    model.eval()
    with torch.no_grad():
        image_transformed_sq = image_transformed_sq.to(device)
        output = model(image_transformed_sq)
        _, predicted_class_index = torch.max(output.data, 1)
    print(f'Predicted Class Index: {predicted_class_index}')
    return predicted_class_index

def calculate_model_performance(y_true, y_pred, class_names):
    num_classes = len(set(y_true + y_pred))
    # build confusion matrix based on predictions and class_index
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for i in range(len(y_pred)):
        # true label on row, predicted on column
        confusion_matrix[y_true[i], y_pred[i]] += 1

    # PER-CLASS METRICS:
    # calculate accuracy, precision, recall, f1 for each class:
    accuracy = torch.zeros(num_classes)
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)
    for i in range(num_classes):
        # find TP, FP, FN, TN for each class:
        TP = confusion_matrix[i, i]
        FP = torch.sum(confusion_matrix[i, :]) - TP
        FN = torch.sum(confusion_matrix[:, i]) - TP
        TN = torch.sum(confusion_matrix) - TP - FP - FN
        # calculate accuracy, precision, recall, f1 for each class:
        accuracy[i] = (TP+TN)/(TP+FP+FN+TN)
        precision[i] = TP/(TP+FP)
        recall[i] = TP/(TP+FN)
        f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
    # calculate support for each class
    support = torch.sum(confusion_matrix, dim=0)
    # calculate support proportion for each class
    support_prop = support/torch.sum(support)

    # OVERALL METRICS
    # calculate overall accuracy:
    overall_acc = torch.sum(torch.diag(confusion_matrix)
                            )/torch.sum(confusion_matrix)
    # calculate macro average F1 score:
    macro_avg_f1_score = torch.sum(f1_score)/num_classes
    # calculate weighted average rF1 score based on support proportion:
    weighted_avg_f1_score = torch.sum(f1_score*support_prop)

    TP = torch.diag(confusion_matrix)
    FP = torch.sum(confusion_matrix, dim=1) - TP
    FN = torch.sum(confusion_matrix, dim=0) - TP
    TN = torch.sum(confusion_matrix) - (TP + FP + FN)

    # calculate micro average f1 score based on TP, FP, FN
    micro_avg_f1_score = torch.sum(
        2*TP)/(torch.sum(2*TP)+torch.sum(FP)+torch.sum(FN))

    # METRICS PRESENTATION
    # performance for each class
    class_columns = ['accuracy', 'precision', 'recall', 'f1_score']
    class_data_raw = [accuracy.numpy(), precision.numpy(),
                      recall.numpy(), f1_score.numpy()]
    class_data = np.around(class_data_raw, decimals=3)
    df_class_raw = pd.DataFrame(
        class_data, index=class_columns, columns=class_names)
    class_metrics = df_class_raw.T

    # overall performance
    overall_columns = ['accuracy', 'f1_mirco', 'f1_macro', 'f1_weighted']
    overall_data_raw = [overall_acc.numpy(), micro_avg_f1_score.numpy(
    ), macro_avg_f1_score.numpy(), weighted_avg_f1_score.numpy()]
    overall_data = np.around(overall_data_raw, decimals=3)
    overall_metrics = pd.DataFrame(
        overall_data, index=overall_columns, columns=['overall'])
    return confusion_matrix, class_metrics, overall_metrics

def get_current_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def save_model_with_timestamp(model, filepath = MODEL_SAVE_LOC):
    filename = get_current_timestamp() + '_cnn_model' + '.pt'
    filepath = os.path.join(filepath, filename)
    torch.save(model.state_dict(), filepath)
    return print('Saved model to: ', filepath)


def save_csv_with_timestamp(train_result_dict, filepath = MODEL_SAVE_LOC):
    filename = get_current_timestamp() + '_training_report' + '.csv'
    filepath = os.path.join(filepath, filename)
    df = pd.DataFrame(train_result_dict)
    df.to_csv(filepath)
    return print('Saved training report to: ', filepath)

def build_csv(directory_string, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    directory_string: string of directory path, e.g. r'.\data\train'
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    import csv
    directory = directory_string
    class_lst = os.listdir(directory) #returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort() #IMPORTANT 
    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #create column names
        for class_name in class_lst:
            class_path = os.path.join(directory, class_name) #concatenates various path components with exactly one directory separator (‘/’) except the last path component. 
            file_list = os.listdir(class_path) #get list of files in class folder
            for file_name in file_list:
                file_path = os.path.join(directory, class_name, file_name) #concatenate class folder dir, class name and file name
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #write the file path and class name to the csv file
    return
freeze_support()
build_csv(train_folder, 'train.csv')
build_csv(test_folder, 'test.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

class_zip = zip(train_df['class_index'], train_df['class_name'])
my_list = []
for index, name in class_zip:
  tup = tuple((index, name))
  my_list.append(tup)
unique_list = list(set(my_list))
print('Training:')
print(sorted(unique_list))
print()

class_zip = zip(test_df['class_index'], test_df['class_name'])
my_list = []
for index, name in class_zip:
  tup = tuple((index, name))
  my_list.append(tup)
unique_list = list(set(my_list))
print('Testing:')
print(sorted(unique_list))

class_names = list(train_df['class_name'].unique())
print(class_names)

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR)
])

train_dataset = FungiDataset(csv_file='train.csv', root_dir="", transform=image_transform)
test_dataset = FungiDataset(csv_file='test.csv', root_dir="", transform=image_transform)
train_dataset, validation_dataset = create_validation_dataset(train_dataset, validation_proportion = 0.2)
print('Train set size: ', len(train_dataset))
print('Validation set size: ', len(validation_dataset))


train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
val_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

# initiation
model = MyCnnModel()
device = get_default_device()
model_prep_and_summary(model, device)
criterion = default_loss()
optimizer = default_optimizer(model = model)
num_epochs = 10

# get training results
trained_model, train_result_dict = train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs)
visualize_training(train_result_dict)
save_model_with_timestamp(trained_model, MODEL_SAVE_LOC)