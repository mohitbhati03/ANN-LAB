import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
step = lambda x: np.where(x >= 0, 1, 0)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = np.tanh
relu = lambda x: np.maximum(0, x)
leaky_relu = lambda x: np.where(x > 0, x, 0.1 * x)
linear = lambda x: x

funcs = [step, sigmoid, tanh, relu, leaky_relu, linear]
titles = ['Step', 'Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'Linear']

plt.figure(figsize=(10, 6))
for i, (f, title) in enumerate(zip(funcs, titles), 1):
    plt.subplot(2, 3, i)
    plt.plot(x, f(x))
    plt.title(title)
    plt.grid()
plt.tight_layout(); 
plt.show()
