# Jeremy Pretty
# CSC 580 Crit 3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.random.set_seed(101)

# Generating random linear data 
x = np.linspace(0, 50, 50).astype(np.float32) 
y = np.linspace(0, 50, 50).astype(np.float32)

# Adding noise to the random linear data 
x += np.random.uniform(-4, 4, 50).astype(np.float32)
y += np.random.uniform(-4, 4, 50).astype(np.float32)

n = len(x)

# Plot the training data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training Data')
plt.show()

# Define TensorFlow placeholders
X = tf.Variable(x)
Y = tf.Variable(y)

# Declare two trainable TensorFlow variables for the Weights and Bias and initializing them randomly using np.random.randn()
W = tf.Variable(np.random.randn(), dtype=tf.float32, name = "W")
b = tf.Variable(np.random.randn(), dtype=tf.float32, name = "b")

learning_rate = 0.001
training_epochs = 1000

# Hypothesis
y_pred = W * X + b

# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

# Gradient Descent Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization():
    with tf.GradientTape() as g:
        pred = W * X + b
        loss = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n)
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Start training
for epoch in range(training_epochs):
    # Run the optimization to update W and b values.
    run_optimization()

    if (epoch+1) % 50 == 0:
        pred = W * X + b
        loss = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n)
        print("Epoch", (epoch + 1), ": cost =", loss.numpy(), "W =", W.numpy(), "b =", b.numpy())

# Calculating the predictions
predictions = W * x + b
print("Training cost =", loss.numpy(), "Weight =", W.numpy(), "bias =", b.numpy(), '\n')

# Plotting the Results
plt.plot(x, y, 'ro', label ='Original data')
plt.plot(x, predictions, label ='Fitted line')
plt.title('Linear Regression Result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


