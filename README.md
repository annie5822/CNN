# CNN
## Simple Handcrafted CNN for MNIST

This project implements a simplified Convolutional Neural Network (CNN) from scratch using **only NumPy**, without relying on deep learning libraries like TensorFlow or PyTorch.

The goal is to train a model to classify handwritten digits (0–9) from the MNIST dataset.


###  Dataset

We use the built-in `mnist.load_data()` from Keras, which provides:

- **Training set**: 60,000 grayscale images (28×28)
- **Test set**: 10,000 grayscale images (28×28)
- **Labels**: Integers from 0 to 9

### Model Architecture

This CNN model contains the following components:

1. **Convolution (3×3 vertical edge kernel)**  
   - Detects vertical features in the image  
   - Output size: 26×26

2. **ReLU Activation**  
   - Applies `max(0, x)` to retain positive activations

3. **Max Pooling (2×2, stride=2)**  
   - Downsamples the feature map  
   - Output size: 13×13

4. **Flattening**  
   - Converts 13×13 feature map into a 169-dimensional vector

5. **Fully Connected Layer (Dense)**  
   - Weight matrix: 10 × 169  
   - Outputs class scores (logits) for 10 digits

6. **Softmax**  
   - Converts logits into class probabilities

7. **Cross Entropy Loss**  
   - Measures how well predicted probabilities match the true labels

### Training Procedure

- Trains for **10 epochs**
- Uses **mini-batch SGD** with batch size = 100
- Learning rate = 0.1
- Parameters (`W` and `b`) are updated manually using gradients

At the end of each epoch, the model is evaluated on **1000 test images** for speed.


