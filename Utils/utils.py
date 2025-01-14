import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import datetime
import os
from Config.loc_config import MODEL_SAVE_LOC
import csv
from torchvision import transforms
import PIL
from torchvision.transforms import v2

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
    Calculate the predicted class indices and their probabilities of the image by the trained model.
    '''
    # Prepare the Image
    image = cv2.imread(image_path)  # Read image using cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform(image)
    
    # Display the transformed image (optional)
    plt.imshow(image_transformed.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)

    # Inference
    model.eval()
    with torch.no_grad():
        image_transformed_sq = image_transformed_sq.to(device)
        output = model(image_transformed_sq)
        
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get the top 5 class indices and their probabilities
        top_probabilities, top_indices = torch.topk(probabilities, k=5)
    
    # Convert to numpy and make the output more readable
    top_probabilities = top_probabilities.cpu().numpy().flatten() * 100  # Convert to percentage
    top_indices = top_indices.cpu().numpy().flatten()

    # Print the results
    for i in range(len(top_indices)):
        print(f'Class Index: {top_indices[i]}, Probability: {top_probabilities[i]:.2f}%')

    return top_indices, top_probabilities

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


def generate_fn_cost_matrix(confusion_matrix):
    # set all elements of cost_matrix to zeros:
    dimension = len(confusion_matrix)
    cost_matrix = torch.zeros(dimension, dimension)
    for j in range(dimension):
        for i in range(dimension):
            cost_matrix[i, j] = confusion_matrix[i, j] / \
                (torch.sum(confusion_matrix[:, j]) - confusion_matrix[j, j])
    # set diagonal back to 0
    for i in range(dimension):
        cost_matrix[i:i+1, i:i+1] = 0
    return cost_matrix


def generate_fp_cost_matrix(confusion_matrix):
    # set all elements of cost_matrix to zeros:
    dimension = len(confusion_matrix)
    cost_matrix = torch.zeros(dimension, dimension)
    for i in range(dimension):
        for j in range(dimension):
            cost_matrix[i, j] = confusion_matrix[i, j] / \
                (torch.sum(confusion_matrix[i, :]) - confusion_matrix[i, i])
    # set diagonal back to 0
    for i in range(dimension):
        cost_matrix[i:i+1, i:i+1] = 0
    return cost_matrix

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


def build_annotation_dataframe(image_location, annot_location, output_csv_name):
    """Builds dataframe and csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    image_location: image directory path, e.g. r'.\data\train'
    annot_location: annotation directory path
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    class_lst = os.listdir(
        image_location)  # returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort()  # IMPORTANT
    with open(os.path.join(annot_location, output_csv_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name',
                        'class_index'])  # create column names
        for class_name in class_lst:
            # concatenates various path components with exactly one directory separator (‘/’) except the last path component.
            class_path = os.path.join(image_location, class_name)
            # get list of files in class folder
            file_list = os.listdir(class_path)
            for file_name in file_list:
                # concatenate class folder dir, class name and file name
                file_path = os.path.join(image_location, class_name, file_name)
                # write the file path and class name to the csv file
                writer.writerow(
                    [file_name, file_path, class_name, class_lst.index(class_name)])
    return pd.read_csv(os.path.join(annot_location, output_csv_name))


def check_annot_dataframe(annot_df):
    class_zip = zip(annot_df['class_index'], annot_df['class_name'])
    my_list = list()
    for index, name in class_zip:
        my_list.append(tuple((index, name)))
    unique_list = list(set(my_list))
    return unique_list


def transform_bilinear(output_img_width, output_img_height):

    policies = [v2.AutoAugmentPolicy.CIFAR10, v2.AutoAugmentPolicy.IMAGENET, v2.AutoAugmentPolicy.SVHN]
    augmenters = [v2.AutoAugment(policy) for policy in policies]

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((output_img_width, output_img_height),
                          interpolation=PIL.Image.BILINEAR),

        transforms.RandomChoice(augmenters),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        # transforms.ColorJitter(brightness=(0,1),contrast=(0,1),saturation=(0,1),hue=(-0.5,0.5)),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return image_transform

def transform_bilinear_validate(output_img_width, output_img_height):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((output_img_width, output_img_height),
                          interpolation=PIL.Image.BILINEAR),


        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return image_transform
