import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from matplotlib.colors import ListedColormap

# Load the dataset
df = pd.read_csv(r'C:\Users\devansh\OneDrive\Documents\Visual code\Python\AI\IRIS.csv')

# Data Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset description:")
print(df.describe())
print("\nChecking for missing values:")
print(df.isnull().sum())

# Convert the 'species' column to numerical categories
df['species'] = pd.factorize(df['species'])[0]

# Data Visualization
print("\nVisualizing pairplot...")
sns.pairplot(df, hue='species', palette='Set2')
plt.show()

print("\nVisualizing correlation heatmap...")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature Scaling
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (K-Nearest Neighbors) with Hyperparameter Tuning
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Model Evaluation
y_pred = grid_search.predict(X_test)

print("\nBest Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score: {:.2f}%".format(grid_search.best_score_ * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualizing Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model: {:.2f}%".format(accuracy * 100))

# Train a separate model on the first two features for visualization
X_train_2d = X_train[:, :2]  # Use only the first two features
X_test_2d = X_test[:, :2]

knn_2d = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
knn_2d.fit(X_train_2d, y_train)

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
                         np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.title('K-NN Decision Boundary (First Two Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundaries for the first two features
print("\nVisualizing decision boundaries...")
plot_decision_boundaries(X_train_2d, y_train, knn_2d)