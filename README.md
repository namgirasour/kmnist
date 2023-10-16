# Hiragana Optical Character Recognition Project

This project utilizes machine learning to develop an optical character recognition (OCR) system for recognizing hiragana characters. A 3-hidden-layer Feedforward Neural Network is constructed and trained on the Kuzushiji-MNIST dataset. The dataset comprises 70,000 images of 28x28 grayscale hiragana characters, specifically focusing on 10 unique characters. 


## Dataset
The Kuzushiji-MNIST dataset is utilized in this project, which contains 70,000 images of 28x28 grayscale hiragana characters. The dataset is divided into 60,000 training images and 10,000 testing images. For this project, the focus is on recognizing 10 unique hiragana characters.

- Dataset source: [Kuzushiji-MNIST Dataset](https://github.com/rois-codh/kmnist)

## Model Architecture
The OCR model is built using a 3-hidden-layer Feedforward Neural Network. The architecture details are as follows:

- Input Layer: 784 neurons (28x28 images flattened)
- Hidden Layer 1: 128 neurons, ReLU activation
- Hidden Layer 2: 64 neurons, ReLU activation
- Hidden Layer 3: 32 neurons, ReLU activation
- Output Layer: 10 neurons, Softmax activation (for 10 unique hiragana characters)

## Performance Metrics
The model has been evaluated based on the following metrics:

- Precision: 93.2%
- Recall: 94%
- Accuracy: 93.3%

