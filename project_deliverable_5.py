#* project_deliverable_4.ipynb
#*
#* ANLY 555 2023
#* Project deliverable 3
#*
#* Due on: 10/27/2023
#* Author(s): Landon Carpenter
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither
#* given nor received any assistance on this project other than
#* the TAs, professor, textbook and teammates.
#*

import csv
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import os
import wordcloud as wordcloud

#create dataset class
class DataSet:
    """
    Class for managing the dataset
    
    Attribute:
        filename (str): the name of the file to be read in
    """

    #constructor
    def __init__(self, filename, ):
        """
        Initializes the DataSet class
        """
        self.filename = filename
        self.data = None
        self.data_type = None

    #create the framework and stubs for __readFromCSV, __load, clean, and explore
    def __readFromCSV(self, filename, header = True):
        """
        Reads in the data from a CSV file. The data is stored on a column basis similar to a parquet file to more easily account for data types.

        Args:
            filename (str): the name of the file to be read in
            header (bool): whether or not the file has a header

        Returns:
            data (np.array): the data read in from the CSV file
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)

                if header:
                    header = data[0]
                    data = data[1:]
                else:
                    header = [f"col_{i}" for i in range(len(data[0]))]

                #init the dict to store the data
                columns = {col_name: [] for col_name in header}

                for row in data:
                    for col_name, value in zip(header, row):
                        try:
                            columns[col_name].append(float(value))
                        except:
                            columns[col_name].append(value)
                            #if the value is '' then replace it with np.nan
                            if value == '':
                                columns[col_name][-1] = np.nan
                            else:
                                pass


                #ok now convert to numpy array
                d_type = [(col_name, object if any(isinstance(val, str) for val in columns[col_name]) else float) for col_name in header]
                self.data = np.array(list(zip(*[columns[col_name] for col_name in header])), dtype = d_type)

        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")

        return self.data


    #abstract base class (ABC)
    def __load(self, filename):
        """
        Loads the data from a CSV file and also calls the getType and readFromCSV functions

        Args:
            filename (str): the name of the file to be read in

        Returns:
            data (np.array): the data read in from the CSV file
        """
        print(f"Loading {filename}...")

        #get the type of data if it is not already set
        if self.data_type is None:
            data_type = self.getType()
        else:
            data_type = self.data_type


        #if the data is time series set header to false
        if data_type == 'time':
            header = False
        else:
            header = True

        self.__readFromCSV(filename, header=header)

        return self.data


    def getType(self):
        """
        This function will be called later in each of the child classes to determine the type of data

        Returns:
            data_type (str): the type of data
        """

        #using while True to avoid infinite loop that I had earlier
        while True:
            data_type = input("Is this data Time Series, Text, Quantitative, or Qualitative? \nPlease type 'Time', 'Text', 'Quantitative', or 'Qualitative':  ")
            #trying to make the prompt a little more forgiving by making the input lowercase and removing whitespace before checking for validity
            norm = data_type.lower().strip()

            #make sure the type is valid
            if norm in ['time', 'text', 'quantitative', 'qualitative']:
                return norm

            #if the type is not valid they will see this message and be prompted to try again
            else:
                print("Please enter a valid data type.")


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

class HeterogeneousDataSet:
    """
    Class for managing a heterogeneous dataset. 
    """

    def __init__(self):
        """
        Initializes the HeterogeneousDataSet class
        """
        self.datasets = []
        self._load_datasets()

    def _load_datasets(self):
        """
        prompt the user for the dataset types they want to load.
        """
        while True:
            #have user specify the file they want to load
            user_file = input("""Please enter the file name you want to load, once you're done type "finished": """)
            
            #if the user types finished then break the loop
            if user_file.lower() == 'finished':
                break
            
            #instaniate the dataset class 
            temp_data = DataSet(user_file)
            #get the type for the loop
            d_type = temp_data.getType()

            if d_type.lower() == 'timeseries':
                self.datasets.append(TimeSeriesDataSet(user_file, d_type))
            elif d_type.lower() == 'text':
                self.datasets.append(TextDataSet(user_file, d_type))
            elif d_type.lower() == 'quantitative':
                self.datasets.append(QuantDataSet(user_file, d_type))
            elif d_type.lower() == 'qualitative':
                self.datasets.append(QualDataSet(user_file, d_type))
            else:
                print("Please enter a valid dataset type.")

    def clean(self):
        """
        Cleans the data
        """
        print("Cleaning Heterogeneous Data Set...")
        for dataset in self.datasets:
            dataset.clean()

    def explore(self):
        """
        Explores the data
        """
        print("Exploring Heterogeneous Data Set...")
        for dataset in self.datasets:
            dataset.explore()

    def select(self, dataset_type):
        """
        Selects a dataset by type

        Args:
            dataset_type (str): the type of dataset to select

        Returns:
            dataset (DataSet): the dataset of the specified type
        """
        for dataset in self.datasets:
            if dataset.__class__.__name__.lower() == dataset_type.lower():
                return dataset
        print(f"Dataset of type {dataset_type} not found.")
        return None

            
        

#use inheritance to create TimeSeriesDataSet class
class TimeSeriesDataSet(DataSet):
    """
    Class for managing the time series dataset. Uses a median filter to clean the data. The data used in the test is mitbih_trian.csv

    Attribute:
        filename (str): the name of the file to be read in
    """

    #constructor
    def __init__(self, filename, data_type):
        """
        Initializes the TimeSeriesDataSet class
        """
        super().__init__(filename)
        self.data_type = data_type
        self.filename = filename
        self.data = self._DataSet__load(filename)

    #override the clean and explore methods from the DataSet class to be specific to the TimeSeriesDataSet class
    def clean(self, filter_size = (3, 3)):
        """
        Cleans the time series data set using a median filter

        Args:
            filter_size (tuple): the size of the filter to use

        Returns:
            filtered_data (np.array): the filtered data
        """
        print("Cleaning Time Series Data Set...")

        #get the stored data that was read in 
        data = self.data

        names = self.data.dtype.names

        arr = []

        for row in data:
            row_arr = []
            for name in names:
                row_arr.append(row[name])

            arr.append(row_arr)
        arr = np.array(arr)

        pad = (filter_size[0] // 2, filter_size[1] // 2)
        print(f"padding: {pad}")
        rows, cols = arr.shape
        print(f"rows: {rows}, cols: {cols}")

        #to store the filtered data
        filtered_data = np.zeros((rows, cols))

        for i in range(rows):
            start, end = i - pad[0], i + pad[0] + 1

            #apply median filter over the row
            filtered_data[i] = np.median(arr[max(start, 0):min(end, rows)], axis = 0)

        self.data = filtered_data


    def explore(self):
        """
        Explores the time series data set by creating at least two visualizations. creates a scatter plot and a line plot.and a saves it to the path that is declared in the function (to visualizations folder). Additionally the user has the option of specifying a title, x-axis label, and y-axis label for the plot.
        """
        print("Exploring Time Series Data Set...")

        #ask the user if they want to specify title, x-axis label, and y-axis label or use defaults
        custom = input("Would you like to specify the title, x-axis label, and y-axis label? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            #get input to ask for the Title, X-axis label, and Y-axis label
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")
        else:
            #set the title, x-axis label, and y-axis label to default values
            title = "ECG Time Series Data"
            x_label = "Time"
            y_label = "Heartbeat"
        
        #create a path to save the plot
        path = 'visualizations/ECG_Time_Series_Scatter.png'
        #set the figure size
        plt.figure(figsize=(10, 5))

        #ask for a row to explore
        row = int(input("Please enter a ID (row) to explore: "))

        #create a scatter plot, I genuinly need this one because I'm still unsure what is going on with this data. 
        #create a scatter plot of the row that was input which would likely be the patients ecg over the give time period (columns)
        plt.scatter(range(len(self.data[row])), self.data[row])

        #set the title, x-axis label, and y-axis label
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        # #show the plot
        # plt.show()

        path = 'visualizations/ECG_Time_Series_Line.png'
        plt.figure(figsize=(10, 5))

        #now create a line plot to show the distribution of the data with less noise
        plt.plot(range(len(self.data[row])), self.data[row])

        #set the title, x-axis label, and y-axis label
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        # #show the plot
        # plt.show()


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


class TextDataSet(DataSet):
    """
    Class for managing the text dataset. 
    """

    def __init__(self, filename, data_type):
        """
        Initializes the TextDataSet class
        """
        super().__init__(filename)
        self.data_type = data_type
        self.filename = filename
        self.data = self._DataSet__load(filename)

    def clean(self):
        """
        Cleans the text data set. Removes stop words and lemmatizes the data.

        Returns:
            cleaned_data (np.array): the cleaned data
        """
        print("Cleaning Text Data Set...")

        #make sure the data loaded 
        if self.data is None:
            print("Error loading data.")
            return

        stop_words = set(stopwords.words('english'))

        #if the column is a string then remove stop words
        for col in self.data.dtype.names:
            if col == 'text':
                col_data = self.data[col]

                for i in range(len(col_data)):
                    word_tokens = nltk.word_tokenize(col_data[i])

                    filtered_sentence = [w for w in word_tokens if not w in stop_words]

                    filtered_sentence = []

                    for w in word_tokens:
                        if w not in stop_words:
                            filtered_sentence.append(w)

                    col_data[i] = filtered_sentence

                self.data[col] = col_data

        #lemmatize the data
        for col in self.data.dtype.names:
            if col == 'text':
                col_data = self.data[col]

                for i in range(len(col_data)):
                    lemmatizer = nltk.stem.WordNetLemmatizer()
                    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in col_data[i]])
                    col_data[i] = lemmatized_output

                self.data[col] = col_data

    def explore(self):
        """
        Explores the text data set. Creates a word cloud and a histogram of the star ratings. The user has the option of specifying a title, x-axis label, and y-axis label for the plot.
        """
        print("Exploring Text Data Set...")

        #ask the user if they want to specify title, x-axis label, and y-axis label or use defaults
        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the word cloud? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            #get input to ask for the Title, X-axis label, and Y-axis label
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")

        else:
            #set the title, x-axis label, and y-axis label to default values
            title = "Word Cloud"
            x_label = ""
            y_label = ""


        path = 'visualizations/word_cloud.png'
        #create a word cloud from the top 100 words
        cloud = wordcloud.WordCloud().generate(self.data['text'][0])
        plt.axis("off")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.imshow(cloud)

        #save the plot
        plt.savefig(path)

        #close the plot
        plt.close()

        # plt.show()

        path = 'visualizations/star_ratings.png'
        plt.figure(figsize=(10, 5))

        #send self.data['stars'] to ints for histogram
        self.data['stars'] = self.data['stars'].astype(int)

        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the histogram? \nPlease type 'Yes' or 'No'.")

        if custom == 'Yes':
            #get input to ask for the Title, X-axis label, and Y-axis label
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")
        
        else:
            #set the title, x-axis label, and y-axis label to default values
            title = "Histogram of Ratings"
            x_label = "Rating"
            y_label = "Frequency"

        #histogram for 1 through 5 star ratings
        plt.hist(self.data['stars'], bins = 5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        #plt.show()



#use inheritance to create QuantDataSet class
class QuantDataSet(DataSet):
    """
    Class for managing the quantitative dataset. 
    """

    #constructor
    def __init__(self, filename, data_type):
        """
        Initializes the QuantDataSet class
        """
        super().__init__(filename)
        self.filename = filename
        self.data_type = data_type
        self.data = self._DataSet__load(filename)

    #override the clean and explore methods from the DataSet class to be specific to the QuantDataSet class
    def clean(self, header = True):
        """
        Cleans the quantitative data set. Replaces missing values with the mean.

        Args:
            header (bool): whether or not the file has a header

        Returns:
            cleaned_data (np.array): the cleaned data
        """
        try: 
            if self.data is None:
                self._DataSet__load(self.filename)

            print("Cleaning Quant Data Set...")

            #iterate by column replacing missing values with the mean
            for col_name in self.data.dtype.names:
                col_data = self.data[col_name]

                #if the data is numeric
                if np.issubdtype(col_data.dtype, np.number):
                    #replace missing values with the mean limited to 2 decimal places
                    col_data[np.isnan(col_data)] = np.round(np.nanmean(col_data), 2)
                    
        except Exception as e:
            print(f"Error cleaning {self.filename}: {str(e)}")

    def explore(self):
        """
        Explores the quantitative data set. Creates a bar plot of the normalized sales for a given product and a bar plot of the total sales for each week. The user has the option of specifying a title, x-axis label, and y-axis label for the plot.
        """
        print("Exploring Quant Data Set...")

        #ask the user for a Product ID to visualize
        product_id = input("Please enter a Product ID from the Quantitative data to visualize (Example: P2): ")

        product_idx = np.where(self.data['Product_Code'] == product_id)[0]

        if len(product_idx) == 0:
            print("Product ID not found.")
            return
        
        row_indx = product_idx[0]


        data_names = []
        for i in range(0, 52):
            data_names.append(f'Normalized {i}')

        vals = []

        for name in data_names:
            vals.append(self.data[row_indx][name])

        #create a bar plot using vals with the data_names as the x-axis labels and the title, x-axis label, and y-axis label as specified by the user if they want
        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the bar plot? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            #get input to ask for the Title, X-axis label, and Y-axis label
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")

        else:
            #set the title, x-axis label, and y-axis label to default values
            title = (f"Normalized Sales | Product ID: {product_id}")
            x_label = "Week"
            y_label = "Normalized Sales"

        path = f'visualizations/Normalized_Sales_{product_id}.png'

        plt.figsize=(10, 6)

        plt.bar(data_names, vals)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #get rid of x-axis labels because they are too long
        plt.xticks([])

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        #plt.show()

        #do the same for non-normalized sales
        data_names = []
        for i in range(0, 52):
            data_names.append(f'W{i}')

        #sum each product to get the total sales for each week
        total_sales = []

        for i in range(0, 52):
            total_sales.append(np.sum(self.data[f'W{i}']))

        #create a bar plot using vals with the data_names as the x-axis labels and the title, x-axis label, and y-axis label as specified by the user if they want
        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the bar plot? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            #get input to ask for the Title, X-axis label, and Y-axis label
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")
        else:
            #set the title, x-axis label, and y-axis label to default values
            title = "Total Sales | Every Product"
            x_label = "Week"
            y_label = "Total Sales"

        path = 'visualizations/Total_Sales.png'

        plt.figsize=(10, 6)

        plt.bar(data_names, total_sales)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #get rid of x-axis labels because they are too long
        plt.xticks([])

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        #plt.show()
        


#use inheritance to create QualDataSet class
class QualDataSet(DataSet):
    """
    Class for managing the qualitative dataset. 
    """

    #constructor
    def __init__(self, filename, data_type):
        """
        Initializes the QualDataSet class
        """
        super().__init__(filename)
        self.filename = filename
        self.data_type = data_type
        self.data = self._DataSet__load(filename)

    #override the clean and explore methods from the DataSet class to be specific to the QualDataSet class
    def clean(self):
        """ 
        Cleans the qualitative data set, replacing missing values in numeric columns with the median and replacing missing values in string columns with the mode. There are a lot of nan values towards the end of the data, I'm not sure if this was how it was intended in the prompt, but I still consider nan a value. So if nan is the most common response empty values will be replaced with nan. Some questions have very few responses. 

        Returns:
            cleaned_data (np.array): the cleaned data
        """
        print("Cleaning Qual Data Set...")

        try:
            for col_name in self.data.dtype.names:
                col_data = self.data[col_name]

                #if the data is numeric
                if np.issubdtype(col_data.dtype, np.number):
                    #replace missing values with the median limited to 2 decimal places
                    col_data[np.isnan(col_data)] = np.round(np.nanmedian(col_data), 2)
                else:
                    #convert any np.nan to 'nan' so mode can be used
                    col_data[col_data != col_data] = 'nan'
                    #replace missing values with the mode
                    mode = max(set(col_data), key = list(col_data).count)
                    col_data[col_data == 'nan'] = mode
                
                self.data[col_name] = col_data
        
        except Exception as e:
            print(f"Error cleaning {self.filename}: {str(e)}")

        try: 
            self.data = self.data[1:]
        except Exception as e:
            print(f"Error cleaning {self.filename}, header error: {str(e)}")
        

                    
    def explore(self):
        """
        Explores the qualitative data set. The user has the option of specifying a title, x-axis label, and y-axis label for the plot. Creates a bar plot of the degrees of the respondents and a bar plot of the first recommended programming language for data science.
        """
        print("Exploring Qual Data Set...")

        #get unique degree from Q4
        degrees = np.unique(self.data['Q4'])

        #get the counts for each degree
        counts = []

        for degree in degrees:
            counts.append(np.sum(self.data['Q4'] == degree))

        #create a bar plot using counts with the degrees as the x-axis labels and the title, x-axis label, and y-axis label as specified by the user if they want
        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the bar plot? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")
        else:
            #set the title, x-axis label, and y-axis label to default values
            title = "Degrees"
            x_label = "Degree"
            y_label = "Frequency"

        #creating labels so the x-axis is not so cluttered
        val_labels = {
            "Doctoral degree": "Doctoral",
            "Master’s degree": "Master’s",
            "Bachelor’s degree": "Bachelor’s",
            "Some college/university study without earning a bachelor’s degree": "Some College",
            "Professional degree": "Professional",
            "No formal education past high school": "High-School",
            "I prefer not to answer": "No Answer",
        }

        x_label_list = [val_labels[degree] for degree in degrees]

        path = 'visualizations/Degrees.png'

        plt.figure(figsize=(10, 10))

        plt.bar(x_label_list, counts)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #fix x ticks and rotate them to 60 degrees
        plt.xticks(rotation = 45)
        #plt.show()

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        #now I want to see what programming languages are recommended, 
        #creating a histogram of Q19
        languages = np.unique(self.data['Q19'])

        #get the counts for each language
        counts = []

        for language in languages:
            counts.append(np.sum(self.data['Q19'] == language))

        #create a bar plot using counts with the languages as the x-axis labels and the title, x-axis label, and y-axis label as specified by the user if they want
        custom = input("Would you like to specify the title, x-axis label, and y-axis label for the bar plot? \nPlease type 'Yes' or 'No'.")
        if custom == 'Yes':
            title = input("Please enter a title for the plot: ")
            x_label = input("Please enter a label for the x-axis: ")
            y_label = input("Please enter a label for the y-axis: ")
        else:
            title = "First Recommended Programming Language for Data Science"
            x_label = "Language"
            y_label = "Frequency"

        path = 'visualizations/First_Recommended_Language.png'

        plt.figure(figsize=(10, 10))

        plt.bar(languages, counts)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #fix x ticks and rotate them to 60 degrees
        plt.xticks(rotation = 45)

        #save the plot
        plt.savefig(path)
        #close the plot
        plt.close()

        #plt.show()
        

#create class for the classifier 
class ClassifierAlgorithm:
    """
    ClassifierAlgorithm ABC class for managing the classifier algorithm
    """
    def __init__(self):
        """
        Initializes the ClassifierAlgotithm class
        """
        pass

    def train(self, train_data, train_labels):
        """
        Trains the classifier algorithm
        """
        print("Not Training in Classifier Algorithm Class...")

    def test(self, test_data, k):
        """
        Tests the classifier algorithm
        """
        print("Not Testing in Classifier Algorithm Class...")

"""
SIMPLE KNN CLASSIFIER COMPLEXITY ANALYSIS

#each line necessary has a comment to the right with the space and time complexity.

Space Complexity:
    S(n) = O(n)

Time Complexity:
    T(n) = O(n^2 log n)

Big O:
    O(n^2 log n)
"""

#create class for simpe KNN that inherets from ClassifierAlgotithm
class simpleKNNClassifier(ClassifierAlgorithm):
    """
    Class for managing the simple KNN classifier
    """
    def __init__(self, k=5):
        """
        Initializes the simpleKNNClassifier class
        """
        super().__init__()
        self.data = None # S(n) T(1)
        self.labels = None # S(n) T(1)
        self.pred_labels = None # S(n) T(1)
        self.k = k # S(1) T(1)

    def train(self, train_data, train_labels):
        """
        Trains the simple KNN classifier

        Args:
            train_data (np.array): the training data
            train_labels (np.array): the training labels
        """
        print("Training simple KNN...")

        self.data = train_data # S(n) T(1)
        self.labels = train_labels # S(n) T(1)

    def test(self, test_data, k):
        """
        Tests the simple KNN classifier
        """
        preds = [] # S(n) T(1)

        for test in test_data:
            #get the distance from the test to each sample
            samples = np.linalg.norm(self.data - test, axis=1) # S(n) T(n) 
            #sort the samples and get indices of the k closest
            sorted_samples = np.argsort(samples)[:self.k] #S(n log n) T(n log n)

            counts = {} # S(c) T(1) c is the number of classes

            #count the number of times each label appears
            for i in sorted_samples:
                label = self.labels[i] #S(n) T(n)
                counts[label] = counts.get(label, 0) + 1 #S(n) T(n)

            #get the most common label
            mode = max(counts, key = counts.get) #S(n) T(n)
            preds.append(mode) #S(n) T(n)

        self.pred_labels = np.array(preds) #S(n) T(1)
        return preds
    
    def pred_knn(self, test_data):
        """
        Use simple KNN to predict the class of a test sample 
        """

        print("Predicting for knn...")

        n_samples = test_data.shape[0] #S(1) T(1)
        n_classes = len(np.unique(self.labels)) #S(1) T(n)
        probs = np.zeros((n_samples, n_classes)) #S(n) T(1)

        for i, sample in enumerate(test_data):
            #get the distance from the test to each training sample
            dist = np.linalg.norm(self.data - sample, axis=1) #S(n) T(n)
            #get k nearest neighbors
            knns = np.argsort(dist)[:self.k] #S(n log n) T(n log n)
            vote_count = np.zeros(n_classes) #S(n) T(1)
            #count the number of times each label appears
            for knn in knns:
                vote_count[self.labels[knn]] += 1 #S(n) T(n)

            #send counts to probs
            probs[i] = vote_count / self.k #S(n) T(1)

        return probs #S(n) T(1)
    
    def get_probs(self, test_data):
        """
        Predicts the probability of a test sample belonging to each class
        """

        print("Predicting probability...")

        probs = self.pred_knn(test_data) #S(n) T(1)
        return probs #S(n) T(1)


class DecisionNode:
    #init method
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, label=None, label_dist=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.label_dist = label_dist

class Tree_str:
    """
    Tree ABC
    """
    def __str__(self):
        pass
    def forTreeVIs(self, node, depth=0, prefix="Root"):
        pass

class decisionTreeClassifier(ClassifierAlgorithm, Tree_str):
    #init method
    def __init__(self, max_depth=3, default_label=0):
        self.max_depth = max_depth
        self.root = None
        self.default_label = default_label
        self.labels = None

    def train(self, training_data, training_labels):
        self.labels = training_labels
        self.unique_labels = np.unique(training_labels)
        self.root = self.make_tree(training_data, training_labels)
        

    def make_tree(self, data, labels, current_depth=0):
        n_samples, n_features = data.shape

        if current_depth >= self.max_depth or n_samples == 0 or len(set(labels)) == 1:
            default_label = self.most_common_label(labels)
            label = self.most_common_label(labels) if n_samples > 0 else default_label
            label_dist = self.calculate_label_distribution(labels)
            return DecisionNode(label=label, label_dist=label_dist)
        
        #get the best split for the data
        best_feature, best_threshold = self.best_split(data, labels)

        #create left and right idxs
        left_idxs = data[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        #deal with empty splits
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            default_label = self.most_common_label(labels)
            label = self.most_common_label(labels) if n_samples > 0 else default_label
            return DecisionNode(label=label)
        
        default_label = self.calculate_label_distribution(labels)

        #create left and right subtrees
        left_sub = self.make_tree(data[left_idxs], labels[left_idxs], current_depth + 1)
        right_sub = self.make_tree(data[~left_idxs], labels[~left_idxs], current_depth + 1)

        return DecisionNode(feature_idx=best_feature, threshold=best_threshold, left=left_sub, right=right_sub)
    
    def calculate_label_distribution(self, labels):
        label_counts = np.bincount(labels, minlength=len(self.unique_labels))
        label_dist = label_counts / label_counts.sum()
        return label_dist
    
    #create the most common label method
    def most_common_label(self, labels):
        if len(labels) == 0:
            return self.default_label
        return np.argmax(np.bincount(labels))
        
    def best_split(self, data, labels):
        n_samples, n_features = data.shape
        #find the best feature and threshold to split the data. Using Ginis impurity as the metric
        best_feature, best_threshold = None, None
        best_gini = 1.0 #worst possible gini index

        n_features = data.shape[1]
        for feature_idx in range(n_features):
            thresholds = np.unique(data[:, feature_idx])
            for threshold in thresholds:
                gini = self.get_gini(data, labels, feature_idx, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def get_gini(self, data, labels, feature_idx, threshold):
        #get the gini index for splitting the data
        left_labels = labels[data[:, feature_idx] < threshold]
        right_labels = labels[data[:, feature_idx] >= threshold]
        #calculate the gini index for the left and right labels
        left_gini = 1 - sum([(np.sum(left_labels == label) / len(left_labels)) ** 2 for label in np.unique(left_labels)]) if len(left_labels) > 0 else 0
        right_gini = 1 - sum([(np.sum(right_labels == label) / len(right_labels)) ** 2 for label in np.unique(right_labels)]) if len(right_labels) > 0 else 0
        #calculate the weighted gini index
        n = len(labels)
        weighted_gini = (len(left_labels) / n) * left_gini + (len(right_labels) / n) * right_gini
        return weighted_gini
    
    def _predict(self, sample, node = None):
        if node is None:
            node=self.root
        if node.label is not None:
            return node.label
        if sample[node.feature_idx] < node.threshold:
            return self._predict(sample, node.left)
        else:
            return self._predict(sample, node.right)
        
    def get_probs(self, test_data):
        """
        Get predicted probs for each class
        """
        print(f'Getting probabilities for decision tree...')
        probs = []
        #get and append the predictions for each sample
        for sample in test_data:
            probs.append(self.get_leaf_dist(sample))

        return np.array(probs)
    
    def get_leaf_dist(self, sample, node=None):
        """
        get the leaf sample distribution for a sample
        """
        #start from root if node is not given
        if node is None:
            node = self.root
        #if None node during recursion
        if node is None:
            return [1.0 / len(np.unique(self.labels))] * len(np.unique(self.labels))
        #leaf
        if node.label is not None:
            return node.label_dist
        #traverse left or right
        if sample[node.feature_idx] < node.threshold:
            return self.get_leaf_dist(sample, node.left)
        else:
            return self.get_leaf_dist(sample, node.right)
        

    def showTree(self, node, depth=0, prefix="Root"):
        """
        Create a string representation of the tree.
        """
        result = ""

        #if the node is a leaf
        if node is not None:
            indent = "  " * depth
            if node.label is not None:
                result += f"{indent}{prefix} - Leaf: [{node.label}]\n"
            else:
                result += f"{indent}{prefix} - [X{node.feature_idx} <= {node.threshold}]\n"
                result += self.printTree(node.left, depth + 1, "Left")
                result += self.printTree(node.right, depth + 1, "Right")
        return result
    
    def forTreeVis(self, node, depth=0, prefix="Root"):
        """
        Print the string representation of the tree. That can be loaded into http://mshang.ca/syntree/ for visualization.
        """
        if node is None:
            return ""
        
        if node.label is not None:
            return f"([{prefix} {node.label}])"
        
        condition = f'X{node.feature_idx} <= {node.threshold}'
        left = self.forTreeVis(node.left, depth + 1, "Left")
        right = self.forTreeVis(node.right, depth + 1, "Right")

        return f"([{prefix} {condition} {left} {right}])"


#create the Experiment class that will run cross validation, get a score given k and, and create a confusion matrix
class Experiment:
    """
    Class for managing the experiment
    """

    def __init__(self, data, labels, classifiers):
        """
        Initializes the Experiment class
        """
        self.data = data
        self.labels = labels
        self.classifiers = classifiers
        self.pred_matrix = np.zeros((len(data), len(classifiers)))

    def ROC(self):
        """
        Generate ROC curve for each classifier. Handling binary and multiclass classification.
        """
        #execute for each classifier
        for classifier in self.classifiers:
            #use the get_probs method to get the probabilities for each class
            probs = classifier.get_probs(self.data)

            #use one vs all approach for > 2 classes
            for class_idx in range(probs.shape[1]):
                actuals = (self.labels == class_idx).astype(int)
                scores = probs[:, class_idx]

                thresholds = np.linspace(0, 1, 100)
                tprs, fprs = [], []
                #get tp, fp, tn, fn for each threshold
                for threshold in thresholds:
                    preds = (scores >= threshold).astype(int)
                    tp = np.sum((actuals == 1) & (preds == 1))
                    fp = np.sum((actuals == 0) & (preds == 1))
                    tn = np.sum((actuals == 0) & (preds == 0))
                    fn = np.sum((actuals == 1) & (preds == 0))

                    tpr = tp / (tp + fn) if tp + fn > 0 else 0
                    fpr = fp / (fp + tn) if fp + tn > 0 else 0

                    tprs.append(tpr)
                    fprs.append(fpr)
                
                sorted_idxs = np.argsort(fprs)
                sorted_fprs = np.array(fprs)[sorted_idxs]
                sorted_tprs = np.array(tprs)[sorted_idxs]

                #plot the ROC curve
                plt.plot(sorted_fprs, sorted_tprs, label=f"{classifier.__class__.__name__} : Class {class_idx}")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("visualizations/ROC.png")


    def runCrossVal(self, folds = 5, CV_k=5):
        """
        Runs k-fold cross validation

        Args:
            k (int): the number of folds to use
        """
        print(f"Running {folds}-fold cross validation...")

        n_samples = len(self.data)
        #get the number of samples per fold
        fold_size = n_samples // folds

        #for each fold train the classifier and test it
        for i in range(folds):
            #get the test data and labels
            test_data = self.data[i * fold_size: (i + 1) * fold_size]
            #dont think I'll need test labels since the prompt has us using creating a score function
            #test_labels = self.labels[i * fold_size: (i + 1) * fold_size]

            #get the training data and labels
            train_data = np.concatenate((self.data[:i * fold_size], self.data[(i + 1) * fold_size:]))
            train_labels = np.concatenate((self.labels[:i * fold_size], self.labels[(i + 1) * fold_size:]))

            for idx, classifier in enumerate(self.classifiers):
                
                #if the classifier isnt a valid classifier then skip it
                if not isinstance(classifier, ClassifierAlgorithm):
                    print(f"Throwing exception for invalid classifier: {classifier.__class__.__name__}")
                    continue
                
                #train the classifier, make preds, and store them in the prediction matrix
                classifier.train(train_data, train_labels)
                preds = classifier.test(test_data, k=CV_k)
                self.pred_matrix[i * fold_size: (i + 1) * fold_size, idx] = preds


    """
    Score Complexity Analysis

    Space Complexity:
        S(n) = O(c) where c is the number of classifiers

    Time Complexity:
        T(n) = O(c*n) same as above

    Big O:
        O(c*n)
    """

    def score(self):
        """
        Scores the experiment
        """
        print("Scoring...")
        #calculate the accuracy score of the preds in pred_matrix, by computing an accuracy score from scratch
        #get the number of correct predictions
        scores = {} # S(c) T(1) c is the number of classifiers
        for classifier in self.classifiers:
            if hasattr(classifier, 'predict_proba'):
                preds = classifier.predict_proba(self.data)
                correct = np.sum(preds == self.labels)
                score = correct / len(self.labels)
                scores[classifier.__class__.__name__] = score
            else:
                print(f'Classifier {classifier.__class__.__name__} cannot score')
        
        #print the results as a table
        print("Scores: ")
        for classifier, acc in scores.items():
            print(f"{classifier}: {acc:.5f}")

    """
    Confusion Matrix Complexity Analysis

    Space Complexity:
        S(n) = O(c) where c is the number of classifiers

    Time Complexity:
        T(n) = O(c*n) same as above

    Big O:
        O(c*n)
    """
    def confusionMatrix(self):
        """
        Creates a confusion matrix
        """
        print("Creating confusion matrix...")
        #initiatiate the matrix with zeros
        for i, classifier in enumerate(self.classifiers):
            tp, fp, tn, fn = 0, 0, 0, 0 # S(1) T(1)

            #get the preds for the classifier
            preds = self.pred_matrix[:, i] # S(n) T(n)

            #get counts for mat
            for j in range(len(preds)):
                if self.labels[j] == 1 and preds[j] == 1: # T(1) this applies to each if statement in this loop
                    tp += 1
                elif self.labels[j] == 0 and preds[j] == 1:
                    fp += 1
                elif self.labels[j] == 1 and preds[j] == 0:
                    fn += 1
                elif self.labels[j] == 0 and preds[j] == 0:
                    tn += 1

            conf_mat = np.array([[tp, fp], [fn, tn]]) # S(1) T(1)

            fig, ax = plt.subplots() # S(1) T(1)
            ax.axis('tight') 
            ax.axis('off')
            ax.table(cellText=conf_mat, colLabels=['Predicted 1', 'Predicted 0'], rowLabels=['Actual 1', 'Actual 0'], loc='center')
            ax.set_title(f"Confusion Matrix for {classifier.__class__.__name__}")

            table = [[tp, fp], [fn, tn]]

            ax.table(cellText=table, colLabels=['Predicted 1', 'Predicted 0'], rowLabels=['Actual 1', 'Actual 0'], loc='center')

            #if a folder called visualizations does not exist then create it
            try:
                os.mkdir('visualizations')
            except:
                print("Couldnt make visualizations folder or it already exists")

            #save the plot
            plt.savefig(f'visualizations/Confusion_Matrix_{classifier.__class__.__name__}.png')

            #close the plot
            plt.close()


#I have created a bash script to ruan all of this automatically so the grader doesn't have to give input for each of the data sets could be annoying when grading multple people's projects

#cleaned data and visualizations are saved to the cleaned_data and visualizations folders respectively

#data is pulled from the data folder

#all paths are refrenced locally

#create a main function to test all of my updates
# if __name__ == "__main__":
#     print("Starting Main...")
    #I'M LEAVING IN THE TEST SCRIPT FOR PART 2. PART 3 IS AT THE BOTTOM

    # paths = {
    #     'time': 'data/Time_ECG.csv',
    #     'text': 'data/Text_Yelp.csv',
    #     'quant': 'data/Quant_Sales.csv',
    #     'qual': 'data/Qual_Survey.csv'
    # }

    # #create a folder called cleaned_data to store the cleaned data from each of the data sets
    # try:
    #     os.mkdir('cleaned_data')
    # except:
    #     print("Couldnt make cleaned_data folder or it already exists")

    # #create a folder called visualizations to store the visualizations from each of the data sets
    # try:
    #     os.mkdir('visualizations')
    # except:
    #     print("Couldnt make visualizations folder or it already exists")

    # print("---Starting Test---")
    # print("Testing Time Series Data Set...")
    # time_data = TimeSeriesDataSet(paths['time'])
    # time_data.clean()
    # np.savetxt('cleaned_data/cleaned_time_data.csv', time_data.data, delimiter=',', fmt='%s')
    # print("First row of cleaned data: ")
    # print(time_data.data[:1])
    # time_data.explore()

    # print('\n')
    # print("Testing Text Data Set...")
    # text_data = TextDataSet(paths['text'])
    # text_data.clean()
    # np.savetxt('cleaned_data/cleaned_text_data.csv', text_data.data, delimiter=',', fmt='%s')
    # print("First row of cleaned data: ")
    # print(text_data.data[:1])
    # text_data.explore()

    # print('\n')
    # print("Testing Quantitative Data Set...")
    # quant_data = QuantDataSet(paths['quant'])
    # quant_data.clean()
    # np.savetxt('cleaned_data/cleaned_quant_data.csv', quant_data.data, delimiter=',', fmt='%s')
    # print("First 1 row of cleaned data: ")
    # print(quant_data.data[:1])
    # quant_data.explore()

    # print('\n')
    # print("Testing Qualitative Data Set...")
    # qual_data = QualDataSet(paths['qual'])
    # qual_data.clean()
    # np.savetxt('cleaned_data/cleaned_qual_data.csv', qual_data.data, delimiter=',', fmt='%s')
    # print("Firs row of cleaned data: ")
    # print(qual_data.data[:1])
    # qual_data.explore()

    # #test the simple KNN classifier
    # print('\n')
    # print("Testing Simple KNN Classifier...")
    # simpleKNN = simpleKNNClassifier()
    # simpleKNN.train(quant_data.data, quant_data.data['Product_Code'])
    # preds = simpleKNN.test(quant_data.data, 5)
    # print("Predictions: ")
    # print(preds)

    # n_train = 15
    # n_test = 5

    # train_data = np.random.randint(1,12, (n_train,3))
    # test_data = np.random.randint(1,12, (n_test,3))

    # train_labels = np.where(train_data.mean(axis=1) > 6, 1, 0)
    # test_labels = np.where(test_data.mean(axis=1) > 6, 1, 0)

    # print(f"---Training Data--- \n {train_data}")
    # print(f"---Training Labels--- \n {train_labels}")
    # print('\n')
    # print(f"---Test Data--- \n {test_data}")
    # print(f"---TestLabels--- \n {test_labels}")

    # #testing the simple KNN classifier
    # print('\n')
    # print("Testing Simple KNN Classifier...")
    # simpleKNN = simpleKNNClassifier()
    # simpleKNN.train(train_data, train_labels)

    # my_experiment = Experiment(train_data, train_labels, [simpleKNN])
    # my_experiment.runCrossVal()

    # my_experiment.score()
    # #show the results from my_experiment
    # print("---Prediction Matrix--- ")
    # print(my_experiment.pred_matrix)

    # #create a confusion matrix
    # #my_experiment.confusionMatrix()