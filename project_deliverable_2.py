#* project_deliverable_2.ipynb
#*
#* ANLY 555 2023
#* Project <>
#*
#* Due on: 10/04/2023
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

        #get the type of data
        data_type = self.getType()

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
            data_type = input("Is this data Time Series, Text, Quantitative, or Qualitative? \nPlease type 'Time', 'Text', 'Quantitative', or 'Qualitative'.")
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
        

#use inheritance to create TimeSeriesDataSet class
class TimeSeriesDataSet(DataSet):
    """
    Class for managing the time series dataset. Uses a median filter to clean the data. The data used in the test is mitbih_trian.csv

    Attribute:
        filename (str): the name of the file to be read in
    """

    #constructor
    def __init__(self, filename):
        """
        Initializes the TimeSeriesDataSet class
        """
        super().__init__(filename)
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


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextDataSet(DataSet):
    """
    Class for managing the text dataset. 
    """

    def __init__(self, filename):
        """
        Initializes the TextDataSet class
        """
        super().__init__(filename)
        self.filename = filename
        self.data = self._DataSet__load(filename)

    def clean(self):
        """
        Cleans the text data set. Removes stop words and lemmatizes the data.

        Returns:
            cleaned_data (np.array): the cleaned data
        """
        print("Cleaning Text Data Set...")

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
    def __init__(self, filename):
        """
        Initializes the QuantDataSet class
        """
        super().__init__(filename)
        self.filename = filename
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
        product_id = input("Please enter a Product ID to visualize: ")

        i = 0
        for row in self.data:
            if row['Product_Code'] == product_id:
                row_indx = i 
            i += 1


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
    def __init__(self, filename):
        """
        Initializes the QualDataSet class
        """
        super().__init__(filename)
        self.filename = filename
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


#I have created a bash script to ruan all of this automatically so the grader doesn't have to give input for each of the data sets could be annoying when grading multple people's projects

#cleaned data and visualizations are saved to the cleaned_data and visualizations folders respectively

#data is pulled from the data folder

#all paths are refrenced locally

#create a main function to test all of my updates
if __name__ == "__main__":
    paths = {
        'time': 'data/Time_ECG.csv',
        'text': 'data/Text_Yelp.csv',
        'quant': 'data/Quant_Sales.csv',
        'qual': 'data/Qual_Survey.csv'
    }

    #create a folder called cleaned_data to store the cleaned data from each of the data sets
    try:
        os.mkdir('cleaned_data')
    except:
        print("Couldnt make cleaned_data folder or it already exists")

    #create a folder called visualizations to store the visualizations from each of the data sets
    try:
        os.mkdir('visualizations')
    except:
        print("Couldnt make visualizations folder or it already exists")

    print("---Starting Test---")
    print("Testing Time Series Data Set...")
    time_data = TimeSeriesDataSet(paths['time'])
    time_data.clean()
    np.savetxt('cleaned_data/cleaned_time_data.csv', time_data.data, delimiter=',', fmt='%s')
    print("First row of cleaned data: ")
    print(time_data.data[:1])
    time_data.explore()

    print('\n')
    print("Testing Text Data Set...")
    text_data = TextDataSet(paths['text'])
    text_data.clean()
    np.savetxt('cleaned_data/cleaned_text_data.csv', text_data.data, delimiter=',', fmt='%s')
    print("First row of cleaned data: ")
    print(text_data.data[:1])
    text_data.explore()

    print('\n')
    print("Testing Quantitative Data Set...")
    quant_data = QuantDataSet(paths['quant'])
    quant_data.clean()
    np.savetxt('cleaned_data/cleaned_quant_data.csv', quant_data.data, delimiter=',', fmt='%s')
    print("First 1 row of cleaned data: ")
    print(quant_data.data[:1])
    quant_data.explore()

    print('\n')
    print("Testing Qualitative Data Set...")
    qual_data = QualDataSet(paths['qual'])
    qual_data.clean()
    np.savetxt('cleaned_data/cleaned_qual_data.csv', qual_data.data, delimiter=',', fmt='%s')
    print("Firs row of cleaned data: ")
    print(qual_data.data[:1])
    qual_data.explore()

