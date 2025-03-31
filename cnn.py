import numpy as np
from tensorflow.keras.datasets import mnist

# ---------- 手刻 CNN 前處理 ----------
def conv2d(input_data, kernel, stride=1):
    H, W = input_data.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            region = input_data[h_start:h_start + kH, w_start:w_start + kW]
            output[i, j] = np.sum(region * kernel)
    return output

def relu(input_data):
    return np.maximum(0, input_data)

def max_pooling(input_data, pool_size=2, stride=2):
    H, W = input_data.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            region = input_data[h_start:h_start + pool_size, w_start:w_start + pool_size]
            output[i, j] = np.max(region)
    return output

def flatten(input_data):
    return input_data.flatten()

# ---------- Dense & Softmax ----------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    eps = 1e-10
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# ---------- 前處理流程：Conv + ReLU + Pool + Flatten ----------
def preprocess_batch(img_batch, kernel):
    N = img_batch.shape[0]
    output = np.zeros((N, 169))  # 每張圖 → 13x13 → flatten 成 169 維
    for i in range(N):
        x = conv2d(img_batch[i], kernel)
        x = relu(x)
        x = max_pooling(x, pool_size=2)
        x = flatten(x)
        output[i] = x
    return output

# ---------- 載入 MNIST ----------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# ---------- 初始化 ----------
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

input_dim = 169  # 13x13 after pooling
output_dim = 10
np.random.seed(0)
W = np.random.randn(output_dim, input_dim) * 0.01
b = np.zeros((output_dim,))

epochs = 10
lr = 0.1
batch_size = 100

# ---------- 訓練 ----------
for epoch in range(epochs):
    perm = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train_oh[perm]

    for i in range(0, len(x_train), batch_size):
        x_batch_img = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # 前處理（CNN 部分）
        x_batch = preprocess_batch(x_batch_img, kernel)

        # Dense + softmax
        logits = np.dot(x_batch, W.T) + b
        probs = softmax(logits)
        loss = cross_entropy(probs, y_batch)

        # 反向傳播
        grad_logits = (probs - y_batch) / batch_size
        grad_W = np.dot(grad_logits.T, x_batch)
        grad_b = np.sum(grad_logits, axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    # 測試準確率
    x_test_processed = preprocess_batch(x_test[:1000], kernel)  # 範例只測 1000 筆加速
    test_logits = np.dot(x_test_processed, W.T) + b
    test_probs = softmax(test_logits)
    test_preds = np.argmax(test_probs, axis=1)
    acc = np.mean(test_preds == y_test[:1000])

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
