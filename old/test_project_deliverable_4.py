import numpy as np
from project_deliverable_4 import decisionTreeClassifier, Experiment, simpleKNNClassifier

def generate_data(n_samples: int = 1000, n_features: int = 2, class_ratio: float = .5):
    X = np.random.randn(n_samples, n_features) * 100
    sums = np.sum(X, axis=1)
    sorted_sums = np.sort(sums)
    thresh_idx = int(n_samples * class_ratio)
    threshold = sorted_sums[thresh_idx]

    y = (sums > threshold).astype(int)

    #print the numbe of samples in each class
    unique, counts = np.unique(y, return_counts=True)
    #print(f'Number of samples in each class - Binary: {dict(zip(unique, counts))}')


    return X, y

def generate_multi_class_data(n_samples: int = 1000, n_features: int = 2, n_classes: int = 3):
    X = np.random.randn(n_samples, n_features) * 100
    weights = np.random.rand(n_features)
    comb = X.dot(weights)

    sorted_features = np.sort(comb)
    thresholds = [sorted_features[int(n_samples / n_classes * (i + 1))] for i in range(n_classes - 1)]

    # Generate class labels based on thresholds
    y = np.digitize(comb, thresholds)

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

    #count and print the number of predictions for each class using the unique function
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f'Simple KNN {dict(zip(unique, counts))}')

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Simple KNN Accuracy: {accuracy}")

def test_decision_tree():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    tree = decisionTreeClassifier(max_depth=5)
    tree.train(X_train, y_train)

    y_pred = [tree._predict(sample) for sample in X_test]

    #count the number of predictions for each class using the unique function
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f'Decision Tree: {dict(zip(unique, counts))})')


    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Decision Tree Accuracy: {accuracy}")

def test_multi_knn():
    X, y = generate_multi_class_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    knn = simpleKNNClassifier(k=5)
    knn.train(X_train, y_train)

    y_pred = knn.test(X_test, k=5)

    #count and print the number of predictions for each class using the unique function
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f'Simple KNN {dict(zip(unique, counts))}')

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Simple KNN Accuracy: {accuracy}")

def test_multi_tree():
    X, y = generate_multi_class_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    tree = decisionTreeClassifier(max_depth=5)
    tree.train(X_train, y_train)

    y_pred = [tree._predict(sample) for sample in X_test]

    #count the number of predictions for each class using the unique function
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f'Decision Tree: {dict(zip(unique, counts))})')


    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Decision Tree Accuracy: {accuracy}")

def test_binary_ROC():
    X, y = generate_data()
    knn = simpleKNNClassifier(k=5)
    knn.train(X, y)
    tree = decisionTreeClassifier(max_depth=5)
    tree.train(X, y)
    exp = Experiment(data=X, labels=y, classifiers=[knn, tree])

    exp.ROC()

def test_multi_ROC():
    X, y = generate_multi_class_data()
    X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    knn = simpleKNNClassifier(k=5)
    knn.train(X_train, y_train)
    tree = decisionTreeClassifier(max_depth=5)
    tree.train(X_train, y_train)

    exp = Experiment(data=X_test, labels=y_test, classifiers=[knn, tree])
    exp.ROC()
    

if __name__ == "__main__":
    test_simple_knn()
    test_decision_tree()
    test_binary_ROC()
    # test_multi_knn()
    # test_multi_tree()
    # test_multi_ROC()