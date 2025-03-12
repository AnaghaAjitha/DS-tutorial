import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_classification

#creates a dataset with 100 points each having 2 features.
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, random_state=42)

#creating a pandas DataFrame to store generated data 
data = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
data['Class'] = y

#divide dataset into training-80% and test-20% data so as to evaluate performance of model; using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scikit-learn lda
lda = LDA(n_components=1)  #reduce to 1D
X_train_lda = lda.fit_transform(X_train, y_train)  #fit and transform
X_test_lda = lda.transform(X_test)  #transform test data

#display the plot-scikit-learn
plt.figure(figsize=(8, 5))#create a figure of 8 inches wide and 5 inches tall
plt.scatter(X_train_lda, y_train, c=y_train, cmap='bwr', edgecolors='k')#scatter plot of lda transformed data
plt.xlabel('LDA Component 1')#x-axis
plt.ylabel('Class')#y-axis
plt.title('LDA Projection (Scikit-Learn)')
plt.show()#display the plot

#lda using matrix operation
#compute class-wise means
mean_0 = np.mean(X_train[y_train == 0], axis=0)
mean_1 = np.mean(X_train[y_train == 1], axis=0)

#compute scatter matrix within class
S_w = np.zeros((X_train.shape[1], X_train.shape[1]))#initialize a zero matrix
#loop through all samples belonging to class 0
for xi in X_train[y_train == 0]:#to select all data points with class label 0
    diff = (xi - mean_0).reshape(-1, 1)
    S_w += diff @ diff.T 
#loop through all samples belonging to class 1
for xi in X_train[y_train == 1]:#to select all data points with class label 1
    diff = (xi - mean_1).reshape(-1, 1)
    S_w += diff @ diff.T 

#compute scatter matrix between class
mean_diff = (mean_1 - mean_0).reshape(-1, 1)
S_b = mean_diff @ mean_diff.T

#compute eigenvalues and eigenvectors for the matrix Sw⁻¹ * Sb
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)

#select the eigenvector corresponding to the largest eigenvalue
best_eigvec = eigvecs[:, np.argmax(eigvals)]

#project training data onto lda component
X_train_lda_manual = X_train @ best_eigvec

#plot lda projection-manual computation
plt.figure(figsize=(8, 5))
plt.scatter(X_train_lda_manual, y_train, c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel('LDA Component 1')#x-axis
plt.ylabel('Class')#y-axis
plt.title('LDA Projection (Matrix Operations)')
plt.show()#display the plot
