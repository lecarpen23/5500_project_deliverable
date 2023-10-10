#* project_deliverable_1.ipynb
#*
#* ANLY 555 2023
#* Project <>
#*
#* Due on: 9/18/2023
#* Author(s): Landon Carpenter
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*

"""
Framework and Stubs for Project Deliverable 1. I will add more documentation for functions as we go along.
"""

#create dataset class
class DataSet:
    """
    Class for managing the dataset
    
    Attribute:
        filename (str): the name of the file to be read in
    """

    #constructor
    def __init__(self, filename):
        """
        Initializes the DataSet class
        """
        self.filename = filename
        self.data = None

    #create the framework and stubs for __readFromCSV, __load, clean, and explore
    def __readFromCSV(self, filename):
        """
        Reads in the data from a CSV file
        """
        print(f"Reading from {filename}...")

    def __load(self, filename):
        """
        Loads the data from a CSV file
        """
        print(f"Loading {filename}...")

    def clean(self):
        """
        Cleans the data
        """
        print("Cleaning...")

    def explore(self):
        """
        Explores the data
        """
        print("Exploring...")

#use inheritance to create TimeSeriesDataSet class
class TimeSeriesDataSet(DataSet):
    """
    Class for managing the time series dataset
    """

    #constructor
    def __init__(self, filename):
        """
        Initializes the TimeSeriesDataSet class
        """
        super().__init__(filename)
    #override the clean and explore methods from the DataSet class to be specific to the TimeSeriesDataSet class
    def clean(self):
        """
        Cleans the time series data set
        """
        print("Cleaning Time Series Data Set...")
    def explore(self):
        """
        Explores the time series data set
        """
        print("Exploring Time Series Data Set...")

#use inheritance to create QuantDataSet class
class QuantDataSet(DataSet):
    """
    Class for managing the quantitative dataset
    """

    #constructor
    def __init__(self, filename):
        """
        Initializes the QuantDataSet class
        """
        super().__init__(filename)
    #override the clean and explore methods from the DataSet class to be specific to the QuantDataSet class
    def clean(self):
        """
        Cleans the quantitative data set
        """
        print("Cleaning Quant Data Set...")
    def explore(self):
        """
        Explores the quantitative data set
        """
        print("Exploring Quant Data Set...")

#use inheritance to create QualDataSet class
class QualDataSet(DataSet):
    """
    Class for managing the qualitative dataset
    """

    #constructor
    def __init__(self, filename):
        """
        Initializes the QualDataSet class
        """
        super().__init__(filename)
    #override the clean and explore methods from the DataSet class to be specific to the QualDataSet class
    def clean(self):
        """ 
        Cleans the qualitative data set
        """
        print("Cleaning Qual Data Set...")
    def explore(self):
        """
        Explores the qualitative data set
        """
        print("Exploring Qual Data Set...")


#create class for the classifier 
class ClassifierAlgotithm:
    """
    Class for managing the classifier algorithm
    """
    def __init__(self):
        """
        Initializes the ClassifierAlgotithm class
        """
        pass

    def train(self):
        """
        Trains the classifier algorithm
        """
        print("Training...")

    def test(self):
        """
        Tests the classifier algorithm
        """
        print("Testing...")

#create class for simpe KNN that inherets from ClassifierAlgotithm
class simpleKNNClassifier(ClassifierAlgotithm):
    """
    Class for managing the simple KNN classifier
    """
    def __init__(self):
        """
        Initializes the simpleKNNClassifier class
        """
        super().__init__()

#create class for kdTree KNN that inherets from ClassifierAlgotithm
class kdTreeKNNClassifier(ClassifierAlgotithm):
    """
    Class for managing the kdTree KNN classifier
    """
    def __init__(self):
        super().__init__()

#create the Experiment class that will run cross validation, get a score given k and, and create a confusion matrix
class Experiment:
    """
    Class for managing the experiment
    """

    def __init__(self):
        """
        Initializes the Experiment class
        """
        pass

    def runCrossVal(self, k):
        """
        Runs k-fold cross validation

        Args:
            k (int): the number of folds to use
        """
        print(f"Running {k}-fold cross validation...")

    def score(self):
        """
        Scores the experiment
        """
        print("Scoring...")

    def __confusionMatrix(self):
        """
        Creates a confusion matrix
        """
        print("Creating confusion matrix...")

#main function: if the py file is ran directly then run the main function and perform the following (rather than importing the file and using a function, for example)
if __name__ == "__main__":
    print("Running main to show framework and stubs...")
    print("\n")
    print("---Creating dataset---")
    dataset = DataSet("my_data.csv")
    dataset._DataSet__readFromCSV("my_data.csv")
    dataset._DataSet__load("my_data.csv")
    dataset.clean()
    dataset.explore()
    print("\n")
    
    print("---Creating TimeSeriesDataSet--- ")
    ts_dataset = TimeSeriesDataSet("my_data.csv")
    ts_dataset._DataSet__readFromCSV("my_data.csv")
    ts_dataset._DataSet__load("my_data.csv")
    ts_dataset.clean()
    ts_dataset.explore()
    print("\n")

    print("---Creating QuantDataSet---")
    quant_dataset = QuantDataSet("my_data.csv")
    quant_dataset._DataSet__readFromCSV("my_data.csv")
    quant_dataset._DataSet__load("my_data.csv")
    quant_dataset.clean()
    quant_dataset.explore()
    print("\n")

    print("---Creating QualDataSet---")
    qual_dataset = QualDataSet("my_data.csv")
    qual_dataset._DataSet__readFromCSV("my_data.csv")
    qual_dataset._DataSet__load("my_data.csv")
    qual_dataset.clean()
    qual_dataset.explore()
    print("\n")

    print("---Creating simple KNN classifiers---")
    simple_knn = simpleKNNClassifier()
    simple_knn.train()
    simple_knn.test()
    print("\n")

    print("---Creating kdTree KNN classifiers--- ")
    kd_tree_knn = kdTreeKNNClassifier()
    kd_tree_knn.train()
    kd_tree_knn.test()
    print("\n")

    print("---Creating experiment--- ")
    experiment = Experiment()
    experiment.runCrossVal(5)
    experiment.score()
    experiment._Experiment__confusionMatrix()