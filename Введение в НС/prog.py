import numpy as np

# сигмоида и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_pr(y):
    return y * (1 - y)

# все 8 входов из таблицы истинности
training_inputs = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
])

# наша функция (x1 XOR x2) AND x3
def target(x):
    x1, x2, x3 = x
    xor = (x1 != x2)
    return 1 if (xor and x3 == 1) else 0

training_outputs = np.array([[target(x)] for x in training_inputs])

# размеры
input_size = 3
hidden_size = 2 # возьмем два нейрона
output_size = 1 # на выход класс 0 или 1
np.random.seed(42)
m1 = 2 * np.random.random((input_size, hidden_size)) - 1
m2 = 2 * np.random.random((hidden_size, output_size)) - 1

for iter in range(50000):
    L0 = training_inputs
    L1 = sigmoid(np.dot(L0, m1))
    L2 = sigmoid(np.dot(L1, m2)) # предсказания
    L2_er = training_outputs - L2
    if iter % 10000 == 0:
        print("iter:", iter, "error:", np.mean(L2_er**2))
    L2_1= L2_er * sigmoid_pr(L2)
    L1_er = L2_1.dot(m2.T)
    L1_1= L1_er * sigmoid_pr(L1)
    m2 += L1.T.dot(L2_1)
    m1 += L0.T.dot(L1_1)

print("Результат на обучающем наборе:")
L1 = sigmoid(np.dot(training_inputs, m1))
L2 = sigmoid(np.dot(L1, m2))
for x, y_true, y_pred in zip(training_inputs, training_outputs, L2):
    print(f"{x}: должно быть {int(y_true[0])}, сеть предсказала {y_pred.item():.6f}")
tests = np.array([
    [0,1,1],
    [1,0,1],
    [1,1,1],
])
print("Проверка:")
for t in tests:
    l1 = sigmoid(np.dot(t, m1))
    l2 = sigmoid(np.dot(l1, m2))
    print(f"{t}: {l2[0]:.6f}")