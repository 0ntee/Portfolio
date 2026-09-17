import numpy as np

class Dense:
    """Полносвязный слой"""
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Инициализация Ксавьера (Xavier) для стабильного обучения
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        self.x = None
        self.dw = None
        self.db = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        z = np.dot(self.x, self.w)
        if self.bias:
            z += self.b
        return z
    
    def backward(self, d_output):
        self.dw = np.dot(self.x.T, d_output)
        if self.bias:
            self.db = np.sum(d_output, axis=0, keepdims=True)
        
        d_input = np.dot(d_output, self.w.T)
        return d_input


class ReLU:
    """Функция активации ReLU с очисткой кэша памяти"""
    def __init__(self):
        self.cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.cache = x > 0.0
        
        activated_output = x * self.cache
        return activated_output

    def backward(self, d_output):
        d_input = d_output * self.cache
        self.cache = None 
        return d_input


class MSELoss:
    """Функция потерь с очисткой кэша памяти"""
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        n_elements = self.y_true.size
        d_output = 2 / n_elements * (self.y_pred - self.y_true)
        
        self.y_pred = None
        self.y_true = None
        return d_output
