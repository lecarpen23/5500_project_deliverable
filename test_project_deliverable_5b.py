import numpy as np
from project_deliverable_5 import LSHKNNClassifier

def make_sample_data(nsamples=250, nfeatures=10, threshold=.5):
    """
    Generate data to test LSHKNNClassifier
    
    Args:
        nsamples (int): number of samples to generate
        nfeatures (int): number of features to generate

    Returns:
        X (np.ndarray): nsamples x nfeatures array of features
        y (np.ndarray): nsamples x 1 array of labels
    """
    data = np.random.rand(nsamples, nfeatures)
    labels = np.sum(data, axis=1) > (nfeatures / threshold)
    labels = labels.astype(int)
    return data, labels

def main():
    data, labels = make_sample_data()

    train_data, test_data = data[:200], data[200:]
    train_labels, test_labels = labels[:200], labels[200:]

    classifier = LSHKNNClassifier()
    classifier.train(train_data, train_labels)

    count = 0
    for i, test_point in enumerate(test_data):
        pred = classifier.test(test_point)
        if pred == test_labels[i]:
            count += 1

    print("Accuracy: {}".format(count / len(test_data)))

if __name__ == "__main__":
    main()