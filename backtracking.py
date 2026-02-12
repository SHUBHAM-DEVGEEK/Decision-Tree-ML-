import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (XOR-like)
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [0, 1, 1, 0]

# Initialize weights and biases
w1 = [[0.1, 0.2],
      [0.3, 0.4]]
b1 = [0.1, 0.1]

w2 = [0.5, 0.6]
b2 = 0.1

learning_rate = 0.5
epochs = 5000

# Training loop
for epoch in range(epochs):
    total_error = 0

    for i in range(len(inputs)):
        x = inputs[i]
        y = outputs[i]

        # -------- Forward Propagation --------
        h1 = sigmoid(x[0]*w1[0][0] + x[1]*w1[1][0] + b1[0])
        h2 = sigmoid(x[0]*w1[0][1] + x[1]*w1[1][1] + b1[1])

        out = sigmoid(h1*w2[0] + h2*w2[1] + b2)

        error = y - out
        total_error += error**2

        # -------- Backpropagation --------
        d_out = error * sigmoid_derivative(out)

        # Update output layer weights
        w2[0] += learning_rate * d_out * h1
        w2[1] += learning_rate * d_out * h2
        b2 += learning_rate * d_out

        # Hidden layer errors
        d_h1 = d_out * w2[0] * sigmoid_derivative(h1)
        d_h2 = d_out * w2[1] * sigmoid_derivative(h2)

        # Update hidden layer weights
        w1[0][0] += learning_rate * d_h1 * x[0]
        w1[1][0] += learning_rate * d_h1 * x[1]
        b1[0] += learning_rate * d_h1

        w1[0][1] += learning_rate * d_h2 * x[0]
        w1[1][1] += learning_rate * d_h2 * x[1]
        b1[1] += learning_rate * d_h2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {total_error:.4f}")

# -------- Testing --------
print("\nFinal Outputs:")
for i in range(len(inputs)):
    x = inputs[i]
    h1 = sigmoid(x[0]*w1[0][0] + x[1]*w1[1][0] + b1[0])
    h2 = sigmoid(x[0]*w1[0][1] + x[1]*w1[1][1] + b1[1])
    out = sigmoid(h1*w2[0] + h2*w2[1] + b2)
    print(f"{x} => {out:.4f}")