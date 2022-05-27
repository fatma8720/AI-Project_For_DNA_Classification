# Import libraries
import sys
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as mpl
import scipy
import os
import pickle
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

#check versions of libraries we will use
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sklearn: {}'.format(skl.__version__))
print('Matplotlib: {}'.format(mpl.__version__))

# load the dataset we will use to build or machine learning model
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
names = ['Class', 'id', 'Sequence']
df = pd.read_csv(url, names=names)

#The Data From the file
print(df)
#a look at a summary of each attribute
print(df.describe())

#to specify only the first sequence
print(df.iloc[0])

# to specify only the sequence "bases" of whole sequence data
print(df['Sequence'].iloc[0])

#to specify only the sequence "class" of whole sequence data
classes = df.loc[:, 'Class']
print(classes)


# Preparing the dataset
# generate list of DNA sequences only
sequences = df.loc[:, 'Sequence']
print(sequences)

#make data set to collect the important and needed data of the imported file
dataset = {}
i = 0

#prepare sequences by spilt the bases and store it distributed to treat it as an independent and each one in its own place as class identifier
# loop through sequences and split into individual nucleotides
for seq in sequences:
    # split into nucleotides, remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in seq if x != '\t']

    # append class assignment
    nucleotides.append(classes[i])

    # add to dataset
    dataset[i] = (nucleotides)

    # increment i
    i += 1

#print dataset lists without make lines between them .. split sequence nucleotides as elements of list and class as the last element of each list

#show the first data in the dataset as a sample of output
print("dataset[0]")
print(dataset[0])

#make the dataset in form of data frame to transform it from sample of list of lists into columns and rows
df = pd.DataFrame(dataset)
print(df)

#make transpose as matrices "make column rows and vice versa" +/- of classes type be as last column instead last row
df = df.transpose()
print(df)

# for clarity,rename the last dataframe column to "class" instead 57
df.rename(columns={57: 'Class'}, inplace=True)

#print it after rename
print(df)

#a look at a summary of each attribute after extract the data
print(df.describe())

# Record value counts for each sequence
series = []
for name in df.columns:
    series.append(df[name].value_counts())

#store the series of the value counts for all columns in our data frame as a data frame also
#when print information data frame about calculated vale counts class will be only in index 57 which is "class type" in last raw of the series
info = pd.DataFrame(series)
print(info)

#make a transpose for data to make the class as a column instead row
details = info.transpose()
print(details)

#Make visualization for our extracted data by histogram and scatter matrix to take a general look
#histogram
info.hist()
pyplot.show()
#scatter
scatter_matrix(info)
pyplot.show()


# We can't run machine learning algorithms on the data in 'String' formats. We need to switch it to numerical data.
# so we will mainplate all sequences by Convert categorical variable [t,c,g,a,+,-] into indicator variables[a0-a56 ,t0-t56 ,...., class+,class-] by using get_dummies function from pandas.
numerical_df = pd.get_dummies(df)
#show these numerical data for our data
print(numerical_df)

# We don't need both class columns,so we will merge it by drop one then rename the other to simply 'Class'
# delete class_-
df = numerical_df.drop(columns=['Class_-'])
# make class + is the base by which if data =1 its + or 0 then -
df.rename(columns={'Class_+': 'Class'}, inplace=True)
#print the final data frame that prepared to be used and divided for tech the machine
print(df)

#show part of the data for one sequence
print(df.iloc[:5])

# Use the model_selection module to separate the prepared data to training and testing datasets
from sklearn import model_selection

# Create X and Y datasets as inputs and outputs for each sequence to pass it to model selection split function to take part as a train and another as a test
#make numerical data of the nucleotides of the sequence as the input
X = np.array(df.drop(['Class'], 1))
print(X)
#make numerical data of the class of the sequence as the output
y = np.array(df['Class'])
print(y)


# define seed for reproducibility "random state"
seed = 1

# split data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=seed)
#show numerical data of the nucleotides of the sequence that model selection create as input train
print(X_train)
#show numerical data of the nucleotides of the sequence that model selection create as input test
print(X_test)
#show numerical data of the class of the sequence that model selection create as output train
print(y_train)
#show numerical data of the class of the sequence that model selection create as input test
print(y_test)

# Now we have our train and test data , we can start building algorithms!
# We'll need to import each algorithm we plan on using from sklearn.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
# We also need to import some performance metrics, such as accuracy_score and classification_report to evaluate models
from sklearn.metrics import classification_report, accuracy_score

# Define models to train
models = []
# store algorithms names and the call functions that perform them
models.append(('Nearest Neighbors', KNeighborsClassifier(n_neighbors=2)))
models.append(('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))))
models.append(('Decision Tree', DecisionTreeClassifier(max_depth=5)))

# evaluate each model in turn
...

# define scoring method
scoring = 'accuracy'
# make list of the results of cross values to make visualization for it
results = []
#store names of each algorithm in model list in list of names to print it
names = []
for name, model in models:
    #split into 10 different smaller sets generally follow the same principles.
    kfold = model_selection.KFold(n_splits=10)
    #pass the model function, our data will learn from it , kfold value will learn with , accuracy as the scoring method
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    #add cross value for each algorithm to results list
    results.append(cv_results)
    #add name of each algorithm to results list
    names.append(name)
    #calculate the mean, standerd deviation and print int
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Visualization of the Comparison Algorithms Cross Values
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Cross Value Comparison')
pyplot.show()


# make list of results for accuracy score to make visualization for it
accuracy_results = []
for name, model in models:
    model.fit(X_train, y_train)
    #estimated output by the machine learning techniques "models" after fed by train data of x,y
    predictions = model.predict(X_test)
    print(name)
    #visualization of estimated and real Y
    pyplot.plot(predictions)
    pyplot.plot(y_test)
    pyplot.title(name)
    pyplot.show()
    #show the accuracy score of each model
    print(accuracy_score(y_test, predictions))
    #add to list to make visualization
    accuracy_results.append(accuracy_score(y_test, predictions))
    #print the classification report about the model behaviour and output compared to the real values
    print(classification_report(y_test, predictions))

# Visualization of the Comparison Algorithms Accuracy Scores
pyplot.bar(names,accuracy_results,width = 0.4)
pyplot.title('Algorithm Accuracy Score Comparison\n')
pyplot.show()