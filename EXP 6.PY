import numpy as np

def bipolar(x):
    return np.where(x == 0, -1, 1)

# Sign activation function
def predict(weights, X):
    s = np.dot(X, weights)
    return np.where(s >= 0, 1, -1)

# Hebbian learning rule with verbose output
def train_hebb_verbose(X, y, lr=0.1, epochs=5):
    w = np.zeros(X.shape[1])   # initialize weights
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        for xi, yi in zip(X, y):
            w += lr * yi * xi   # Hebb’s update rule
            print(f"Input: {xi}, Target: {yi}, Updated Weights: {w}")
        preds = predict(w, X)
        preds01 = np.where(preds == -1, 0, 1)
        print(f"Predictions after epoch {epoch}: {preds01.tolist()}")
    return w

# Truth table inputs with bias term (bias = 1)
X = np.array([
    [1, 0, 0],  # bias, x1, x2
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# -------------------- User Input --------------------
choice = input("Enter gate (AND/OR): ").strip().upper()

if choice == "AND":
    targets = np.array([0, 0, 0, 1])
elif choice == "OR":
    targets = np.array([0, 1, 1, 1])
else:
    print("Invalid choice! Please enter AND or OR.")
    exit()

# Convert targets to bipolar
y = bipolar(targets)

# Train
final_w = train_hebb_verbose(X, y, lr=0.2, epochs=5)

print("\nFinal Weights:", final_w)
