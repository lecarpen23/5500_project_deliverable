import numpy as np
import matplotlib.pyplot as plt
from project_deliverable_4 import DecisionTreeClassifier, Experiment, simpleKNNClassifier

def generate_test_data(n_samples: int = 500, n_features: int = 2):
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y

def main():
    #use generate_test_data to generate a dataset
    X, y = generate_test_data()

    #set decision_tree and knn_classifier the methods I created in project_deliverable_4.py
    decision_tree = DecisionTreeClassifier(max_depth=3)
    knn_classifier = simpleKNNClassifier(k=5)

    #set up my_experiment and run it using cross validation and then show the ROC, score, and confusion matrix
    my_experiment = Experiment(data=X, labels=y, classifier=[decision_tree, knn_classifier])

    #print statements like this help me debug and I think they look nice
    print(f"Running experiment")
    print(f'Running cross validation')
    my_experiment.runCrossVal(folds=5)
    print(f'Running ROC')
    my_experiment.ROC()
    print(f'Running score')
    my_experiment.score()
    print(f'Running confusion matrix')
    my_experiment.confusion_matrix()
    print(f'Creating Decision Tree Visualization')
    print(decision_tree)

    print(f"Finished running experiment")

if __name__ == "__main__":
    main()





