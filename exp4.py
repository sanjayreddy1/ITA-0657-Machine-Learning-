import numpy as np

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed
np.random.seed(1)

# Initialize weights
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))

# Training
epochs = 10000
lr = 0.1

for epoch in range(epochs):

    # Forward Pass
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

# Testing
print("Final Output after Training:")
print(final_output)