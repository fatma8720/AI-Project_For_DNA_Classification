# AI-Project_For_DNA_Classification


Table of content
	OVERVIEW OF THE PROJECT 
	Project Environment
	Feature Selection
	Training the models with Data
	the Class is predicted 
	Result
	Introduction about the problem
	Machine learning algorithms used

	METHDOLOGY AND EXPERIMENTAL SIMULATION
	START INSTALL, IMPORT PYTHON LIBRARIES AND CHECK VERSIONS
	LOAD DATA AND STORE IT IN ACCESSIBLE WAY FOR WORK
	PREPARE THE DATASET
	CONTINUE PREPARE THE DATASET
	VISUALIZATION FOR EXTRACTED DATA
	UNIVARIATE PLOTS
	MULTIVARIATE PLOTS
	SWITCH EXTRACTED DATA TO NUMERICAL DATA 
	TRAINING AND TESTING DATA 
	START BUILDING ALGORITHMS 
	EVALUATE EACH MODEL IN TURN

	RESULTS AND DISCUSSION
	Classification Report and Accuracy score Discussion
	Comparison between Cross Values 
	Visualization of estimated and Test Y
	For k-Nearest Neighbors
	For Gaussian Process
	For Decision Process
	Comparison between Accuracy Scores
	Visualized by Bar Plots

	CONCLUSION AND RECOMMENDATIONS
	REFERANCES
	APPENDIX


OVERVIEW OF THE PROJECT 
Project Environment

-	We used PyCharm as the coding platform and downloaded the DNA dataset from the UCI repository. K-Nearest Neighbor (K-NN) and other classification approaches are used in our methodology.

 Feature Selection

-	Identifying the subset of original features using various methods based on the information provided, accuracy, and prediction mistakes.
	The features used in the project are:
	a (0-56) (e.g.: 0_a, 1_a, ..., 56_a)
	c (0-56) (e.g.: 0_a, 1_a, ..., 56_a)
	g (0-56) (e.g.: 0_a, 1_a, ..., 56_a)
	t (0-56) (e.g.: 0_a, 1_a, ..., 56_a)
	Class (0/1)


 Model Selection

	 Nearest Neighbors
	 Decision Tree
	 Gaussian Process


Training the models with Data
	The data source that we used:
	https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data
 

TRAINING DATA AND TESTING DATA. 
	 Training data accounts for 80% of the data, whereas testing data accounts for 20%.


the Class is predicted 
	‘+'(1).
	'-’ (0).


Methodology step
1.	Installing the Python and SciPy platform.
2.	Loading the dataset.
3.	Summarizing the dataset.
4.	Visualizing the dataset.
5.	Evaluating some algorithms.
6.	Making some predictions.


Result 
	 Accuracy on Testing set:
	Nearest Neighbors - 86.36363636363636 %
	Decision Tree - 86.36363636363636 %
	Gaussian Process - 95.45454545454546 %






INTRODUCTION ABOUT THE PROBLEM
Grasp organisms requires an understanding of DNA. In all species, DNA contains most of the genetic instructions for development, function, and reproduction. We can now quickly read a DNA sequence thanks to advances in sequencing technology. According to data from the National Human Genome Research Institute's Genome Sequencing Program, the cost of reading one million base pairs dropped from over $5000 in September 2001 to $0.014 in October 2015. The amount of data on DNA sequences is also growing at an exponential rate. In December 2015, for example, the size of GenBank, a popular database of DNA sequences, has surpassed 2 billion base pairs. It would be fantastic if we could combine these massive data sets with the computing capability of today's computers to aid our understanding of DNA.
We employed a machine learning model to classify DNA in this experiment, which calculates information about each nucleotide concentration in sequences and its class type, such as:
	Nearest Neighbors
	Decision Tree
	Gaussian Process
By applying to these models, we determined the accuracy of each of them, and on this basis, we came out with the most accurate classification, and it was the “Gaussian Process "with the most Accuracy of 95.45454545454546% against other models [Nearest Neighbors with 86.36363636363636%, and Decision Tree with 86.36363636363636].
MACHINE LEARNING ALGORITHMS USED 
K-nearest Neighbors
K-nearest neighbors (k-NN) is a pattern recognition technique that finds the k closest relatives in future cases using training datasets.
When using k-NN for classification, you calculate to place data within the category of its nearest neighbor. If k = 1, it will be assigned to the class closest to 1. K is classified by a plurality poll of its neighbors.
	Complexity: o(n).

Decision Tree
A decision tree is a supervised learning algorithm that is perfect for classification problems because it can order classes at a precise level. It works like a flow chart, separating data points into two similar categories at a time, starting with the "tree trunk" and progressing to "branches" and "leaves," where the categories become more finitely similar. This results in the formation of categories within categories, allowing for organic classification with minimal human supervision.

	The random forest algorithm is an extension of the decision tree algorithm in that you build many decision trees with training data, and then fit your new data into one of the trees as a "random forest."
	It, essentially, averages your data to connect it to the nearest tree on the data scale. Random forest models are helpful as they remedy for the decision tree’s problem of “forcing” data points within a category unnecessarily.
	Complexity O (log n).
 
Gaussian Process
By performing one-versus-one based training and prediction, the Gaussian Process Classifier supports multi-class classification. For each pair of classes, one binary Gaussian process classifier is fitted and trained to distinguish these two classes. These binary predictors' predictions are combined to form multi-class predictions. 
It may be less expensive computationally because it must solve many problems involving only a subset of the entire training set rather than fewer problems on the entire dataset. Because Gaussian process classification scales cubically with dataset size, this could be much faster. It should be noted, however, that "one vs one" does not support predicting probability estimates, only straight predictions. Furthermore, the Gaussian Process Classifier does not (yet) internally implement a true multi-class Laplace approximation.
	Complexity: o(n^3).




 METHODOLOGY AND EXPERIMENTAL SIMULATION     
1. Start Install, Import Python Libraries and Check Versions

      To make sure that our versions match what we will work with or be more recent.

	Explanation and Pseudocode:
	If Libraries:
- do not exist in your python:
        then install it 
- else 
              skip
	print versions you have 
	from printed data, check if it matches your needs, new or not:
i.	if it matches 
      then skip 
ii.	else 
     install suitable ones




2. Load Data and store it in an accessible way for work

      To make sure that our versions match what we will work with or be more recent.

	Explanation and Pseudocode:
	load the dataset we will use to build or machine learning model:
   with pandas “read_csv” function
	Store data from the file in the data frame and show it
	look at a summary of each attribute
      with the “describe” function for data frames
	Do some access specifications to reach specific data int it to know how to deal with such as views only: 
-	Only classes 
-	Only sequences 
-	Only first sequence data




3. Prepare the Dataset

	Explanation and Pseudocode:
o	make a dataset list [] to collect the important and needed data of the imported file
o	prepare sequences by spilt the bases and storing them distributed to treat them as an independent and each one in its own place as class identifier
	data set <- []
	I <- 0
	for seq do:
-	nucleotides <- list(seq)
-	J<- 0
-	for seq[J] do:
	check if seq[J]! = “\t”:
                                 - nucleotide <- seq[J]
-	nucleotides <- nucleotide + class[I]
-	dataset <- nucleotides
-	I <- I+1

o	show the first data in the dataset as a sample of output
o	make the dataset, in form of a data frame to transform it from a sample of a list
   df,<- pd.DataFrame(dataset)


          

	

3. Continue to Prepare the Dataset

o	Make transpose as matrices "make column rows and vice versa" +/- of classes type be as the last column instead the last row
       df <- df. transpose ()
o	For clarity, rename the last data frame column to "class" instead of 57
     df. rename (columns= {57: 'Class'}, inplace=True)
o	Show it after renaming and a look at a summary of each attribute after extracting the data with “df. describe ()”
o	Record value counts for each sequence
    series <- []
for name in df. columns … Do:
    series <- series+ (df[name].value_counts ()).
o	Make a statical summary for data 
      df.describe().
o	Store the series of the value counts for all columns in our data frame as a data frame also
      info <- pd. DataFrame(series).
  
o	When print information data frame about calculated vale counts class Will be only in index 57 which is "class type" in last raw of the series, so make a transpose for data to make the class as a column instead of row 
     details<-info. Transpose ().

	




4. Visualization for extracted data
	Explanation and Pseudocode:
o	By histogram and scatter matrix to take a general look 


Univariate Plots
	histogram 
o	info.hist()
o	pyplot.show()
              


Multivariate Plots
	scatter
o	scatter_matrix(info)
o	pyplot.show()



5. switch extracted data to numerical data 
	Explanation and Pseudocode:
o	We can't run machine learning algorithms on the data in 'String' formats. We need to switch it to numerical data, so we will manipulate all sequences by Converting categorical variable [t, c, g, a, +, -] into indicator variables [a0-a56, t0-t56, class +, class-] by using get_dummies function from pandas.   
    numerical_df = pd.get_dummies(df)

o	We don't need both class columns, so we will merge them by dropping one then rename the other to simply 'Class'
	delete class_-
   df <- numerical_df. drop(columns=['Class_-'])

	make class + is the base by which if data =1 its + or 0 then –
df. rename (columns= {'Class_+': 'Class'}, inplace=True)

o	Show the final data frame that is prepared to be used and divided for tech the machine.
o	Use the model_selection module to separate the prepared data into training and testing datasets
      from sklearn import model_selection

o	Create X and Y datasets as inputs and outputs for each sequence to pass it to the model selection split function to take part as a train and another as a test
	make numerical data of the nucleotides of the sequence as the input
                           X <-np. array (df. drop(['Class'], 1))
	make numerical data of the class of the sequence as the output
                            y <- np. array(df['Class'])



6. Training and Testing Data 
	Explanation and Pseudocode:
o	Define seed for reproducibility "random state"
                    seed <- 1
o	Split data into training and testing datasets
                    X_train, X_test, y_train, y_test = model_selection. 
                       . train_test_split (X, y, test_size=0.20, random_state=seed)
o	Show numerical data of the nucleotides of the sequence that model selection creates as input train
                     print(X_train)
o	Show numerical data of the nucleotides of the sequence that model selection creates as input test
                     print(X_test)
o	Show numerical data of the class of the sequence that model selection creates as output train
                     print(y_train)
o	Show numerical data of the class of the sequence that model selection creates as input test
                     print(y_test)

7. Start building Algorithms 
	Explanation and Pseudocode:
o	We'll need to import each algorithm we plan on using from sklearn
	from sklearn. neighbors import KNeighboursClassifier
	from sklearn.gaussian_process import GaussianProcessClassifier
	from sklearn.gaussian_process.kernels import RBF
	from sklearn.tree import DecisionTreeClassifier
o	We also need to import some performance metrics, such as accuracy score s and classification reports to evaluate models.
	from sklearn. Metrics import classification_report, accuracy_score
o	Define models to train:
	Identify list to store data about models 
                    models <- []
	Store algorithms names and the call functions that perform them
•	models.append(('Nearest Neighbors', KNeighborsClassifier(n_neighbors=2)))
•	models.append(('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))))
•	models.append(('Decision Tree', DecisionTreeClassifier(max_depth=5)))





8. Evaluate each model in turn
	Explanation and Pseudocode:
o	Initialize variable scoring to 'accuracy'
                            scoring <-' accuracy' 
o	Initialize list of results of cross values to make visualization for it
                             results <- []
o	Initialize list names to store names of each algorithm in a model list in the names list
                               names<- []
o	for names, model in models …DO:
	Split into 10 different smaller sets generally follow the same principles
                                kfold <- model_selection.KFold(n_splits=10)
	Pass the model function, our data will learn from it, kfold value will learn with, accuracy as the scoring method 
    cv_results <- model_selection.cross_val_score(model, X_train, y_train,          cv=kfold, scoring=scoring)
	Add cross value for each algorithm to results list
                                   results.append(cv_results) 

	Add the name of each algorithm to the results list.
                            names. Append(name)
	calculate the mean, and standard deviation and print it 
                                      msg = "%s: %f (%f)" % (name, cv_results. Mean (), cv_results.std ())

o	Make Visualization of the Comparison Algorithms Cross Values
                       pyplot.boxplot(results, labels=names)
                       pyplot.title('Algorithm Cross Value Comparison')
                       pyplot.show()



	Explanation and Pseudocode:
o	Make a list of results for the accuracy score to make a visualization for it
                     accuracy_results = [] 	
o	For name, model in models
•	Learn the machine and the mechanism of classification of the data
	model.fit (X_train, y_train)
•	Estimated output by the machine learning techniques "models" after fed by train data of x, y
	predictions = model. Predict (X_test)
•	Print the name of the model
	print(name)
•	Visualization of estimated and real Y
	Representation of both predicted and actual value for Y_test in the Visualization
-	pyplot.plot(predictions)
-	pyplot.plot(y_test)
	Add a title
-	pyplot.title(name)
	Show the Visualization
-	pyplot.show()



•	Show the accuracy score of each model
	print (accuracy score (y_test, predictions))
•	Add to list to make visualization
	accuracy_results. Append (accuracy score (y_test, predictions))
•	Show the classification report about the model behaviour and output compared to the real values
	print (classification_report (y_test, predictions))
•	Visualization of the Comparison Algorithms Accuracy Scores
	Representation of both names, accuracy_results test in the Visualization
-	pyplot.bar(names,accuracy_results,width = 0.4)
	Add a title
-	pyplot.title('Algorithm Accuracy Score Comparison\n')



RESULTS AND DISCUSSION

i.	From the accuracy score of each algorithm, we will see that:
	Accuracy on Testing sets:
	Nearest Neighbors - 86.36363636363636 %
	Decision Tree - 86.36363636363636 %
	Gaussian Process - 95.45454545454546 %
           The best one is the Gaussian process  


ii.	From the classification report of each algorithm, we will see that:
	Precision: This means How much the accuracy of the correct decision taken and the ratio of true positives.
         [True Positives/(True Positives + False Positives)]
we will see that precision for:
o	Nearest Neighbors – [0.93, 0.75]
o	Gaussian Process – [1.00,0.88]
o	Decision Tree – [1.00,0.54]
1.	Then the highest values for the Gaussian process also

	Recall(sensitivity): Which mean the ratio of true to all the words that were actually false.
      [True Positives/ (True Positives + False Negatives)]
 we will see that Recall for:
o	Nearest Neighbors – [0.87, 0.86]
o	Gaussian Process – [93.00,1.00]
o	Decision Tree – [0.60,1.00]
                     2. Then the highest values for the Gaussian process also


	Accuracy: it of F1 Score: number of times in which the model predicts the y test correctly true prediction
which = 2 * ((*Precision* * *Recall*) / (*Precision* + *Recall*))
o	f1-score: weighted harmonic means of precision and recall such that the best score is 1.0 and the worst is 0.0 and lower than accuracy
o	we will see that f1-score for:
•	Nearest Neighbors – [0.90, 0.80]
•	Gaussian Process – [0.97,0.93]
•	Decision Tree – [0.70,0.75]
                                            3. Then the highest values for the Gaussian process also
	we will see that accuracy of F1-score for:
•	Nearest Neighbors – [0.86]
•	Gaussian Process – [0.95]
•	Decision Tree – [0.73]
                                           4. Then the highest values for the Gaussian process also


	Support: Which a number of samples in each class is:
       5. equivalent for all models and it is for accurate results.

Then from the previous discussion of the evaluation models of classification report and accuracy score content for each algorithm and from that are supported in the following visualization of the results we will see that the most accurate model of machine learning models we use to classify data is the Gaussian process model

Classification Report

Comparison between Cross Values 
	We will see that almost all values of the Gaussian process plot at the top of all cross values plots of another model

Visualization of estimated and Test Y For k-Nearest Neighbors
Visualization of estimated and Test Y For Gaussian Process
Visualization of estimated and Test Y For Decision Process

Comparison between Accuracy Scores Visualized by Bar Plots




CONCLUSION 
We chose to use machine learning to classify DNA, so we followed the seven steps:

o	Gathering Data 
	Data is a file that contains 106 sequences, each sequence has a class and id
o	Preparing Data
	The file is converted into a data frame, giving each column in the file a name
o	Choose model
	We used 3 models to work on the data
•	k nearest neighbor algorithm
•	Gaussian process algorithm
•	Decision tree algorithm
o	Training and Testing
	Training data accounts for 80% of the data, whereas testing data accounts for 20%0.
o	Hyperparameter
	used in processes to help estimate model parameters.
o	Prediction
	In which algorithms predict the output y-test from the input X-test as we fed it with train data X and y.
o	Evaluation
	We used accuracy and cross-validation
	The result of Comparing the actual values with the estimated values is defined as the best accuracy
	Through this step, we show the accuracy of the models as follows
•	Nearest Neighbors - 86.36363636363636 %
•	Decision Tree - 86.36363636363636 %
•	Gaussian Process - 95.45454545454546 %
•	The determination that the best model for our data is the one that has the highest accuracy, in our code is Gaussian Process with an accuracy 95.45454545454546 %



Our recommendations for the future, in general, are to try to keep pace with future changes in machine learning and to try to understand, learn and use it. Especially about our project is to try to understand it and if it can be developed using different algorithms like:
•	Random Forest 
•	Neural Net 
•	AdaBoost 
•	Naive Bayes 
•	SVM Linear 
•	SVM RBF 
•	SVM Sigmoid 
•	SVM Polynomials



REFERANCES
https://www.biologydiscussion.com/dna/classification/classification-of-dna-7-criterias-with-diagram/15458
https://archive.ics.uci.edu/ml/index.php
API Reference — scikit-learn 1.1.1 documentation
https://scikit-learn.org/stable/modules/neighbors.html
https://scikit-learn.org/stable/modules/gaussian_process.html
https://scikit-learn.org/stable/modules/tree.html


APPENDIX
https://github.com/fatma8720/AI-Project_For_DNA_Classification


