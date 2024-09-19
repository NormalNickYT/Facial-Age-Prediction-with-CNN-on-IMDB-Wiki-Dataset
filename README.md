# Facial-Age-Prediction-with-CNN-on-IMDB-Wiki-Dataset

## Project Overview

This repository contains a Convolutional Neural Network (CNN) model for facial age prediction using the IMDB-Wiki dataset. The model is designed to classify facial images into two categories: "Younger than 41" and "Older than 41". 

## Introduction

This project uses a CNN to predict the age category of individuals based on their facial images. The CNN is built using TensorFlow and Keras and has been trained on a subset of the IMDB-Wiki dataset.

## Dataset

The dataset used in this project is the IMDB-Wiki dataset, which consists of facial images along with age labels. For this project, images are categorized into two classes based on whether the individual is younger or older than 41 years.

You can download it here: 

You can download the dataset here: [IMDB-Wiki Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## Model Architecture

The model architecture includes several layers of Separable Convolutions followed by MaxPooling and BatchNormalization layers:

1. **Input Layer**: The image where all different sizes so i resized it the input takes Images of size 128 x 128 and 3 channels (RGB).
2. **Convolutional Layers**:
   - SeparableConv2D layers with increasing filters (64, 128, 256) and ReLU activation.
   - MaxPooling2D layers to reduce spatial dimensions.
   - SpatialDropout2D layers to prevent overfitting.
   - BatchNormalization layers to normalize activations.
3. **Fully Connected Layers**:
   - Dense layers with ReLU activation and dropout for regularization.
   - Final Dense layer with a softmax activation functio.

## Training

The model was trained using the following settings:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy but you could use binary crossentropy here
- **Metrics**: Accuracy
- **Data Augmentation**: ImageDataGenerator with rotation, shift, shear, zoom, and horizontal flip.

Training was performed on the IMDB-Wiki dataset, with images resized to (128, 128) and divided into training and validation sets.

## Predictions

The trained model is used to predict the age category of facial images. Predictions are made by:
1. Loading and preprocessing images.
2. Passing the images through the trained CNN model.
3. Outputting the predicted probability of the image belonging to each category.

The model achieved an accuracy of 75% on the validation set.
