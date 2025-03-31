# CNN
conv2d(input_data, kernel, stride=1)
Function: Simulates 2D convolution operation in CNN.

Parameters:

input_data: A single grayscale image (28×28).

kernel: The convolution filter (3×3 in this case).

stride: Step size, default is 1.

Returns: A feature map after convolution (26×26).

relu(input_data)
Function: ReLU activation function.

Purpose: Applies max(0, x) element-wise to keep positive values and suppress negatives (introduces non-linearity).

max_pooling(input_data, pool_size=2, stride=2)
Function: Simulates max pooling.

Purpose: Takes the maximum value in each 2×2 region to reduce dimensionality and retain important features. (26×26 ➜ 13×13)

flatten(input_data)
Function: Flattens a 2D array into a 1D vector.
(13×13 ➜ 169-dimensional vector)

Dense Layer & Loss Function
softmax(x)
Function: Outputs class probabilities.

Mechanism: Applies softmax row-wise so that each row sums to 1, representing probability distribution over classes.

cross_entropy(y_pred, y_true)
Function: Computes the cross-entropy loss between predicted probabilities and true one-hot labels.

Purpose: Used as the loss function during training (lower is better).

one_hot(y, num_classes=10)
Function: Converts integer class labels into one-hot encoded vectors.
(e.g., 3 ➜ [0 0 0 1 0 0 0 0 0 0])

Full Preprocessing Pipeline
preprocess_batch(img_batch, kernel)
Function: Applies the CNN preprocessing steps to a batch of images, converting each into a 169-dimensional vector.

Steps:

conv2d ➜ output shape: 26×26

relu

max_pool ➜ output shape: 13×13

flatten ➜ final shape: 169-dimensional vector
