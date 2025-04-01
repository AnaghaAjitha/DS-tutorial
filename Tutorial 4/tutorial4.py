import numpy as np  # Import NumPy for numerical operations
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor from sklearn
from sklearn.utils import resample  # Import resample for bootstrap sampling
from sklearn.datasets import make_regression  # Import make_regression to generate a synthetic dataset

class RandomForestRegressorCustom:
    def __init__(self, n_estimators=10, max_features='sqrt', max_depth=None):
        # Initialize the random forest with the number of trees, max features, and depth limit
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []  # List to store decision tree models
        self.feature_indices = []  # List to store selected feature indices for each tree
    
    def fit(self, X, y):
        self.trees = []  # Clear previous trees
        self.feature_indices = []  # Clear previous feature selections
        n_samples, n_features = X.shape  # Get number of samples and features
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling: randomly select data with replacement
            X_sample, y_sample = resample(X, y, random_state=np.random.randint(0, 10000))
            
            # Select a random subset of features based on max_features setting
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))  # Square root of total features
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))  # Log base 2 of total features
            else:
                max_features = n_features  # Use all features if not specified
            
            feature_indices = np.random.choice(n_features, max_features, replace=False)  # Random feature selection
            self.feature_indices.append(feature_indices)  # Store selected feature indices
            
            # Train a Decision Tree on the sampled data with the selected features
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)  # Store trained tree
    
    def predict(self, X):
        predictions = np.zeros((self.n_estimators, X.shape[0]))  # Store predictions from each tree
        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_indices)):
            predictions[i] = tree.predict(X[:, feature_indices])  # Predict using the selected features for each tree
        return np.mean(predictions, axis=0)  # Return the average prediction from all trees

# Example usage
# Generate dataset
X, y = make_regression(n_samples=5, n_features=3, noise=0.1)  # Create synthetic regression dataset

# Train model
rf = RandomForestRegressorCustom(n_estimators=3, max_features='sqrt', max_depth=5)  # Initialize random forest
rf.fit(X, y)  # Train the model

# Predict
predictions = rf.predict(X)  # Make predictions
print(predictions)  # Print predictions
