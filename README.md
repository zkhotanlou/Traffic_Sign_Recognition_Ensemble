# Traffic Sign Recognition using Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Implementation](#implementation)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architectures](#model-architectures)
  - [Stacked Ensemble Model](#stacked-ensemble-model)
  - [Training](#training)
- [Results](#results)
  

## Introduction

In today's world, neural networks and deep learning are used in various fields and have proven effective in addressing a wide range of challenges and issues. One of the areas where deep learning has found application is in the domain of autonomous vehicles, where automobiles need to autonomously perceive their surrounding environment. Consequently, one of the sub-challenges that can be defined is the recognition of road signs and driving instructions by vehicles, enabling them to autonomously recognize traffic rules in their environment. In this project, we aim to solve this problem using various neural networks, each with its own specific strengths. When these networks are combined, they can complement each other's weaknesses, leading to superior results. In this project, five networks have been selected, and their predictions have been stacked together, achieving an outstanding 100% result.

## Implementation

### Dataset

The dataset used in this project consists of images of 43 road signs, totaling 51,839 images. These images were converted into matrix data and split into 70% for training, 10% for validation, and 20% for testing. To avoid performing these operations on the data each time, they were saved as pickle files.

### Data Preprocessing

Data preprocessing involves data augmentation to make the networks less sensitive to image variations, including rotation and cropping. Histogram equalization is also applied for data normalization.

### Model Architectures

The project explores several neural network architectures, including:
- Convolutional Neural Network (CNN)
- MobileNet
- DenseNet121
- ResNet50
- EfficientNet

#### CNN
The network typically starts with a two-dimensional Convolution layer, processing 32x32 pixel input images with three color channels (RGB). This layer is followed by a ReLU activation function and a MaxPooling layer, which aids in feature detection, such as identifying edges. These blocks are repeated multiple times with varying input dimensions to allow for progressive feature extraction. To classify images into 43 different traffic sign classes, a dense layer with a softmax activation function is added to the network.
The data is then compiled using the Adam optimizer and the Sparse Categorical Cross-Entropy loss function. Training is performed with a batch size of 64 and 20 epochs on the training data.  
<img width="163" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/61290e61-cffc-4363-a7c5-0fcce901ca1f">  


#### MobileNet  
MobileNet is a subset of CNN networks, designed initially for use in mobile vision systems, and is aimed at maintaining network accuracy while reducing the number of trainable parameters. MobileNet uses separable convolutions, significantly reducing the number of parameters compared to conventional CNNs with the same depth. MobileNet employs two types of convolutional operations:  
**Depthwise Convolution:** A depthwise convolution applies a convolution filter for each input channel. In a regular 2D convolution applied over multiple input channels, the filter is the same size as the input, allowing channels to be freely combined to produce each element in the output.  
**Pointwise Convolution:** A pointwise convolution uses a 1x1 kernel, which is a kernel of size 1x1 and has depth for each input channel.

In the code provided, we define an initial MobileNet model suitable for our dataset's image dimensions, which has previously been trained on the ImageNet dataset. First, we add the primary MobileNet network, followed by a "Flatten" layer to convert the matrix inputs to one-dimensional arrays. Then, a Dense layer with ReLU activation and an output size of 512 is added, followed by a dropout layer to randomly deactivate some neurons to prevent overfitting. Finally, another dense layer is added to perform classification into the 43 classes defined for traffic signs.  
<img width="277" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/09b811e6-58d7-4644-8d35-4790fe5c2ac0">  


#### DenseNet121 Model
In a standard Convolutional Neural Network (CNN), every convolutional layer except the first one (which takes the input) receives its input from the previous convolutional layer and produces a feature map that is then passed to the next convolutional layer. Hence, for n layers, there are n direct connections between each layer and the next layer, with each connection representing an n x n connectivity. This leads to the problem of vanishing gradients, as the gradient magnitude significantly decreases during backpropagation, and thus, rarely any changes are made to the weights. To solve this issue, DenseNets modify the standard CNN architecture and simplify the connections between layers. In the DenseNet architecture, every layer is directly connected to every other layer, and the network is known as a Densely Connected Convolutional Network (DenseNet). For n layers, there are 1/2 * n * (n + 1) connections.  
<img width="218" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/d0bb1325-f7f1-4313-b104-90afab0cd178">  

The implementation of this model is similar to the previous ones, with minor differences in hyperparameter values. The lower number of epochs compared to MobileNet is due to the lower number of trainable parameters in the network.  
<img width="210" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/26a22e63-c3d7-4c1d-a60e-9b8f859a3b0f">  


#### ResNet50 Model
Convolutional Neural Networks (CNNs) suffer from a significant problem called gradient vanishing. During backpropagation, the gradient magnitude reduces considerably, leading to infrequent updates in weights. To address this issue, the ResNet model uses a technique called skip connections. A skip connection is a direct connection that skips some of the layers in the network. As a result, the output is not the same, as in a regular connection. Without skip connections, the input X is multiplied by the layer's weights, followed by a bias term, and then the activation function is applied to the output. With skip connections, the output is as follows: F(X) + X, where F(X) is the result of the convolution and weight multiplication.

In ResNet50, two types of blocks are used: **identity blocks and convolutional blocks**. The 'X' value is added to the output if and only if the input size matches the output size. If they don't match, a convolutional block is added to the shortcut path to adjust the input size to match the output size.  
<img width="238" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/8aaa2893-8175-43be-9e8f-ab4f1283aa78">  

The implementation of this model is similar to the previous ones, with minor differences in the addition of layers to the initial network.    
<img width="208" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/4d9a1d0e-3d7d-4c0f-8462-834eef0721aa">  


#### EfficientNet Model
EfficientNet is a convolutional neural network architecture and a scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional methods that arbitrarily scale these factors, the EfficientNet scaling method uniformly increases the network's width, depth, and resolution 2^k times with a set of constant scaling coefficients. For example, if we want to use n times more computational resources, we can easily increase the network's depth by α, width by β, and image size by γ. These constant coefficients are determined by searching a small network in the original model. EfficientNet uses a compound coefficient to scale the width, depth, and resolution of the network uniformly in a principled way. The compound scaling method is justified by the observation that if the input image is larger, the network needs more layers to increase the receptive field and more channels to capture finer-grained patterns on the larger image.

The implementation of this model is similar to previous models and only adds a dense layer with a sigmoid activation function to the original network to ensure the output dimensions match the number of defined classes for this task.  
<img width="341" alt="image" src="https://github.com/zkhotanlou/Traffic_Sign_Recognition_Ensemble/assets/84021970/2d43c94c-af60-4168-b4fc-513489e36998">  

#### Stacked Ensemble Model
A stacked ensemble model using pre-trained networks was implemented to enhance the project goal. Models are saved in .h5 format to avoid retraining. The models are loaded, and predictions are made probabilistically. These predictions are then stacked together using, a tensor containing predictions from all initial models.
The 'model_stack_fit' function processes these predictions. It predicts the output classes probabilistically and trains a logistic regression model with stacked tensor and 'inputy'. Additionally, a function determines the most frequently predicted class by the 5 models, ensuring accurate ensemble predictions.

### Training

Each model is trained using the training dataset and evaluated using the validation dataset. The results, including accuracy and loss, are recorded for analysis.

## Results

- **CNN Model**:
  - Training Accuracy: 0.99, Loss: 0.0142
  - Validation Accuracy: 0.5274, Loss: 0.97
  - Test Accuracy: 0.96, Loss: 1.2261

- **MobileNet Model**:
  - Training Accuracy: 0.98, Loss: 0.0632
  - Validation Accuracy: 0.95, Loss: 0.2184
  - Test Accuracy: 0.95, Loss: 0.2570

- **DenseNet121 Model**:
  - Training Accuracy: 0.99, Loss: 0.0192
  - Validation Accuracy: 0.95, Loss: 0.2856
  - Test Accuracy: 0.95, Loss: 0.2466

- **ResNet50 Model**:
  - Training Accuracy: 0.98, Loss: 0.0944
  - Validation Accuracy: 0.92, Loss: 0.5364
  - Test Accuracy: 0.92, Loss: 0.6191

- **EfficientNet Model**:
  - Training Accuracy: 0.99, Loss: 0.0077
  - Validation Accuracy: 0.91, Loss: 0.5206
  - Test Accuracy: 0.92, Loss: 0.4565
    
- **Ensemble Model**:
  The stacked ensemble model combines the results from all the individual models and achieves a perfect accuracy of 1.00.
 

