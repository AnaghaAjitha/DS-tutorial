import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for visualization

# Define the SVM class
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  # Learning rate for gradient descent updates
        self.lambda_param = lambda_param  # Regularization parameter to prevent overfitting
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector, initialized later
        self.b = None  # Bias term, initialized later

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get the number of samples and features
        
        # Ensure labels are either -1 or 1 for correct hinge loss computation
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weight vector with zeros
        self.w = np.zeros(n_features)
        self.b = 0  # Initialize bias to zero
        
        # Perform gradient descent for n_iters iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Compute if the data point satisfies the margin condition
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # If correctly classified, apply only regularization update
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If misclassified, update weight and bias using hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
        
        # Print learned parameters
        print("Learned weights (w):", self.w)
        print("Learned bias (b):", self.b)
    
    def predict(self, X):
        # Compute the linear decision function
        linear_output = np.dot(X, self.w) + self.b
        # Classify based on sign (returns -1 or 1)
        return np.sign(linear_output)

# Generate synthetic dataset
np.random.seed(1)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)  # Linearly separable classes

# Train SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    # Create grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour for decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')

# Plot the decision boundary
plt.figure()  # Create a new figure
plot_decision_boundary(X, y, svm)
plt.show()  # Show the plot

# Another figure (e.g., you can plot other visualizations or data)
plt.figure()  # Create a second figure
plt.plot(np.arange(1, 11), np.random.randn(10), label="Random Data")
plt.title("Another Plot")
plt.legend()
plt.show()  # Display second plot
