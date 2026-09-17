import numpy as np
# Импортируем модули из modules.py
from modules import Dense, ReLU, MSELoss

if __name__ == "__main__":
    np.random.seed(35)

    # Исходные данные
    X = np.array([[1.0, 2.0, -1.0], 
                  [0.5, -1.0, 2.0]])

    Y_true = np.array([[1.0, 0.0], 
                       [0.0, 1.0]])

    # Инициализация графа вычислений
    layer = Dense(in_features=3, out_features=2, bias=True)
    activation = ReLU()
    loss_fn = MSELoss()

    epochs = 200
    learning_rate = 0.1

    print("Запуск процесса обучения...\n")

    for epoch in range(1, epochs + 1):
        # ПРЯМОЙ ХОД (Forward Pass)
        a1 = layer(X)
        y_pred = activation(a1)
        loss = loss_fn(y_pred, Y_true)

        # ОБРАТНЫЙ ХОД (Backward Pass)
        d_loss = loss_fn.backward()
        d_act = activation.backward(d_loss)
        layer.backward(d_act)
        
        # Обновление весов (Градиентный спуск)
        layer.w -= learning_rate * layer.dw
        layer.b -= learning_rate * layer.db

        # Логирование процесса обучения
        if epoch % 20 == 0 or epoch == 1:
            print(f"Эпоха {epoch:03d} | Текущий loss: {loss:.6f}")
            
    print("\nОбучение завершено успешно!")
