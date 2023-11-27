from project_deliverable_5 import HeterogeneousDataSet, QualDataSet, QuantDataSet, TextDataSet, TimeSeriesDataSet, DataSet

"""I need to change the logic, because it only has the file name and then calls getType(), but its passing two arguments """

def main():
    print("Testing HeterogeneousDataSet class: use any of the files in the data folder... \n")

    #instantiate the class
    h_data = HeterogeneousDataSet()

    if not h_data.datasets:
        print("Failed to load datasets or none given.")
        return
    
    print("Cleaning datasets...")
    h_data.clean()

    print("Exploring datasets...")
    h_data.explore()

if __name__ == "__main__":
    main()
