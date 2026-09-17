# Custom Deep Learning Framework Components from Scratch

A clean, modular implementation of essential neural network components built completely from scratch using **NumPy**. This project demonstrates a deep understanding of core deep learning mechanics, vectorization, and the mathematics of backpropagation without relying on high-level frameworks like PyTorch or TensorFlow.

## Key Features

* **Modular Architecture**: Mimics modern deep learning frameworks (like PyTorch `nn.Module`) using isolated components with structured `forward` and `backward` passes.
* **Vectorized Computing**: Avoids explicit Python `for` loops during training, utilizing NumPy matrix operations for maximum execution efficiency.
* **Explicit Memory Management**: Explicitly cleans up cached activation masks and target references (`self.cache = None`) right after gradient computation to prevent memory leaks during long training loops.
* **Xavier Initialization**: Implements proper weight initialization to stabilize gradients and prevent early convergence saturation.

## Implemented Components

* **`Dense` Layer**: Fully-connected linear layer supporting dynamic input/output feature mapping, bias integration, and gradient computation for both weights (`dw`), biases (`db`), and inputs (`d_input`).
* **`ReLU` Activation**: Rectified Linear Unit activation function featuring automated boolean masking and backpropagation caching.
* **`MSELoss`**: Mean Squared Error cost function tracking model predictions vs. ground truth targets with automatic array reference clearing post-backward pass.

## Pipeline Showcase

The framework includes a standalone test pipeline simulating binary classification outputs across a feature matrix:

* **Input Data (X)**: Matrix of size 2 × 3 (2 objects, 3 features each)
* **Targets (\(Y_{true}\))**: Ground truth matrix of size 2 × 2
* **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of `0.1`

### Execution Output
During 200 epochs of training, the model successfully backpropagates errors and drives the Mean Squared Error down toward zero:

```text
Эпоха 001 | Текущий loss: 1.258312
Эпоха 020 | Текущий loss: 0.161042
Эпоха 040 | Текущий loss: 0.038411
...
Эпоха 200 | Текущий loss: 0.000104
```

## Requirements & Installation

No external heavy libraries needed. Just pure Python and NumPy.

1. Clone the repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install numpy
   ```
3. Run the training script:
   ```bash
   python perceptron.py
   ```
