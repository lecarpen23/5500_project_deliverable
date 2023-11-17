class simpleKNNClassifier(ClassifierAlgorithm):
    # ... existing code ...

    def predict_proba(self, test_data):
        """
        Predicts probabilities for the test data using the kNN algorithm.
        """
        num_samples = test_data.shape[0]
        num_classes = len(np.unique(self.labels))
        probabilities = np.zeros((num_samples, num_classes))
        
        for i, test_point in enumerate(test_data):
            # Calculate distances from the test point to all training points
            distances = np.linalg.norm(self.data - test_point, axis=1)
            # Find the k-nearest neighbors
            k_nearest_neighbors = np.argsort(distances)[:self.k]
            # Count the class occurrences among the neighbors
            class_votes = np.zeros(num_classes)
            for neighbor_index in k_nearest_neighbors:
                class_votes[self.labels[neighbor_index]] += 1
            # Convert counts to probabilities
            probabilities[i] = class_votes / self.k
        
        return probabilities

class Experiment:
    # ... existing methods ...
    
    def ROC(self):
        """
        Generates and plots the ROC curve for each classifier.
        It assumes that the classifiers have a method predict_proba that returns
        the probability scores for the positive class.
        """
        plt.figure()
        
        for idx, classifier in enumerate(self.classifiers):
            # Get predicted probabilities
            predicted_probabilities = classifier.predict_proba(self.data)
            # Assume positive class is labeled as '1'
            positive_class_probabilities = predicted_probabilities[:, 1]
            
            # Calculate true positive rates and false positive rates
            fpr, tpr, _ = self.compute_fpr_tpr(self.labels, positive_class_probabilities)
            roc_auc = np.trapz(tpr, fpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{classifier.__class__.__name__} (AUC = {roc_auc:.2f})')
        
        # Plot formatting
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.show()
    
    def compute_fpr_tpr(self, true_labels, positive_probabilities):
        """
        Computes the false positive rate and true positive rate for various threshold settings.
        """
        # Sort by probability scores descending
        sorted_indices = np.argsort(positive_probabilities)[::-1]
        sorted_labels = true_labels[sorted_indices]
        
        tpr = []
        fpr = []
        
        # Compute thresholds by iterating through sorted probabilities
        for threshold_idx, threshold in enumerate(sorted_labels):
            # Apply threshold
            thresholded_predictions = (positive_probabilities >= threshold).astype(int)
            # Compute confusion matrix elements
            tp = np.sum((thresholded_predictions == 1) & (sorted_labels == 1))
            fp = np.sum((thresholded_predictions == 1) & (sorted_labels == 0))
            tn = np.sum((thresholded_predictions == 0) & (sorted_labels == 0))
            fn = np.sum((thresholded_predictions == 0) & (sorted_labels == 1))
            
            # Calculate rates
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
        
        return np.array(fpr), np.array(tpr), sorted_indices

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class decisionTreeClassifier(ClassifierAlgorithm):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def train(self, training_data, training_labels):
        self.root = self.build_tree(training_data, training_labels)

    def build_tree(self, data, labels, current_depth=0):
        num_samples, num_features = data.shape
        # Get the unique labels in the current node
        unique_labels = np.unique(labels)
        
        # Stopping conditions
        if len(unique_labels) == 1 or current_depth == self.max_depth:
            label = unique_labels[0]
            return DecisionNode(label=label)
        
        # Find the best split
        best_feature, best_threshold = self.best_split(data, labels, num_samples, num_features)
        
        # Grow the children recursively
        if best_feature is not None:
            left_indices = data[:, best_feature] < best_threshold
            left_node = self.build_tree(data[left_indices], labels[left_indices], current_depth + 1)
            right_node = self.build_tree(data[~left_indices], labels[~left_indices], current_depth + 1)
            return DecisionNode(feature_index=best_feature, threshold=best_threshold, left=left_node, right=right_node)
        else:
            # Return the most common label if no split is found
            label = max(unique_labels, key=list(labels).count)
            return DecisionNode(label=label)

    def best_split(self, data, labels, num_samples, num_features):
        # This function finds the best feature and threshold to split on
        # For simplicity, it uses the Gini index as the criterion
        # Implementing a full-fledged algorithm is beyond the scope here
        best_feature, best_threshold = None, None
        best_gini = 1.0  # Maximum possible value
        
        for feature_index in range(num_features):
            thresholds = np.unique(data[:, feature_index])
            for threshold in thresholds:
                gini = self.calculate_gini(data, labels, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def calculate_gini(self, data, labels, feature_index, threshold):
        # This function calculates the Gini index for a split on a given feature and threshold
        # It's a measure of impurity - lower is better
        left_labels = labels[data[:, feature_index] < threshold]
        right_labels = labels[data[:, feature_index] >= threshold]
        # Calculate the gini for the left and right subsets
        left_gini = 1.0 - sum([(np.sum(left_labels == label) / len(left_labels))**2 for label in np.unique(left_labels)])
        right_gini = 1.0 - sum([(np.sum(right_labels == label) / len(right_labels))**2 for label in np.unique(right_labels)])
        # Weighted average of the left and right gini
        weighted_gini = (len(left_labels) / len(labels)) * left_gini + (len(right_labels) / len(labels)) * right_gini
        return weighted_gini

    def predict(self, test_data):
        predictions = [self._predict(test_point) for test_point in test_data]
        return np.array(predictions)

    def _predict(self, data_point, node=None):
        if node is None:
            node = self.root
        if node.label is not None:
            return node.label
        if data_point[node.feature_index] < node.threshold:
            return self._predict(data_point, node.left)
        return self._predict(data_point, node.right)

    def __str__(self):
        # Here we would implement a way to visually represent the tree
        # This is a complex task, so for now we'll just return a placeholder string
        return "Decision tree structure"

# Example usage:
dt_classifier = decisionTreeClassifier(max_depth=3)
dt_classifier.train(train_data, train_labels)
predictions = dt_classifier.predict(test_data)
