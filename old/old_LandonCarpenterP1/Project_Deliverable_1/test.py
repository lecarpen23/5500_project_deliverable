import project_deliverable_1
from project_deliverable_1 import DataSet
from project_deliverable_1 import TimeSeriesDataSet
from project_deliverable_1 import QuantDataSet
from project_deliverable_1 import QualDataSet
from project_deliverable_1 import ClassifierAlgotithm
from project_deliverable_1 import simpleKNNClassifier
from project_deliverable_1 import kdTreeKNNClassifier
from project_deliverable_1 import Experiment

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