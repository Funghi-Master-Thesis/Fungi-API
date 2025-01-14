# models.py
from enum import Enum
import timm
from torchvision import models
import torch.nn as nn

class ModelName(Enum):
    RESNET50 = "ResNet50"
    ViTB16 = "vit_b_16"
    Levit128s = "levit_128s"
    DenseNet = "DenseNet"
    VGG16 = "VGG16"
    

class DataSetName(Enum):
    DataSetLast2Days = "DataSetLast2Days"
    DataSetSig = "DataSetSig"
    DataSetCutLast2Days = "DataSetCutLast2Days"


class ResNet50:
    def __init__(self, num_classes):
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def get_model(self):
        return self.model

class ViTB16:
    def __init__(self, num_classes):
        self.model = models.vit_b_16(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.heads[-1].in_features, num_classes) ## change parameters to work with the model

    def get_model(self):
        return self.model
    
class Levit128s:
    def __init__(self, num_classes):
        self.model = timm.create_model('levit_128s.fb_dist_in1k', pretrained=True, num_classes=0) # remove classifier nn.Linear
        
    def get_model(self):
        return self.model
    
class DenseNet:
    def __init__(self, num_classes):
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def get_model(self):
        return self.model
    
class VGG16:
    def __init__(self, num_classes):
        self.model = models.vgg16(weights='IMAGENET1K_V1')
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)   
    def get_model(self):
        return self.model 

                              
def get_model(model_name, num_classes):
    if model_name == "ResNet50":
        return ResNet50(num_classes).get_model()
    elif model_name == "vit_b_16":
        return ViTB16(num_classes).get_model()
    elif model_name == "levit_128s":
        return Levit128s(num_classes).get_model()
    elif model_name == "DenseNet":
        return DenseNet(num_classes).get_model()
    elif model_name == "VGG16":
        return VGG16(num_classes).get_model()
    else:
        raise ValueError(f"Model {model_name} is not supported.")