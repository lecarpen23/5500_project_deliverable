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
    #constructor
    def __init__(self, filename):
        self.filename = filename

    #create the framework and stubs for __readFromCSV, __load, clean, and explore
    def __readFromCSV(self, filename):
        print(f"Reading from {filename}...")

    def __load(self, filename):
        print(f"Loading {filename}...")

    def clean(self):
        print("Cleaning...")

    def explore(self):
        print("Exploring...")

#use inheritance to create TimeSeriesDataSet class
class TimeSeriesDataSet(DataSet):
    #constructor
    def __init__(self, filename):
        super().__init__(filename)

#use inheritance to create QuantDataSet class
class QuantDataSet(DataSet):
    #constructor
    def __init__(self, filename):
        super().__init__(filename)

#use inheritance to create QualDataSet class
class QualDataSet(DataSet):
    #constructor
    def __init__(self, filename):
        super().__init__(filename)

#create class for the classifier 
class ClassifierAlgotithm:
    def __init__(self):
        pass

    def train(self):
        print("Training...")

    def test(self):
        print("Testing...")

#create class for simpe KNN that inherets from ClassifierAlgotithm
class simpleKNNClassifier(ClassifierAlgotithm):
    def __init__(self):
        super().__init__()

#create class for kdTree KNN that inherets from ClassifierAlgotithm
class kdTreeKNNClassifier(ClassifierAlgotithm):
    def __init__(self):
        super().__init__()

#create the Experiment class that will run cross validation, get a score given k and, and create a confusion matrix
class Experiment:
    def __init__(self):
        pass

    def runCrossVal(self, k):
        print(f"Running {k}-fold cross validation...")

    def score(self):
        print("Scoring...")

    def __confusionMatrix(self):
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