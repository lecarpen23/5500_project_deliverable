import numpy as np
from project_deliverable_4 import decisionTreeClassifier, Experiment, simpleKNNClassifier

np.random.seed(42)

def generate_data(n_samples: int = 500, n_features: int = 2, threshold: float = 250):

    X = np.random.randn(n_samples, n_features) * 100
    y = np.sum(X, axis=1) > threshold
    y = y.astype(int)

    return X, y

def my_train_test_split(X, y, test_size=.2):
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def test_simple_knn():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    knn = simpleKNNClassifier(k=5)
    knn.train(X_train, y_train)

    y_pred = knn.test(X_test, k=5)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Simple KNN Accuracy: {accuracy}")

def test_decision_tree():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    tree = decisionTreeClassifier(max_depth=5)
    tree.train(X_train, y_train)

    y_pred = [tree._predict(sample) for sample in X_test]

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Decision Tree Accuracy: {accuracy}")

if __name__ == "__main__":
    test_simple_knn()
    test_decision_tree()