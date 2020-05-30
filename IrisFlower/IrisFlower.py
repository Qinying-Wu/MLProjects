#Qinying Wu
#This is the first machine learning project - Iris Flower
#Tutorial privided by https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
#Commenced on May 30, 2020

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "../iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#pandas.read_csv(filepath_or_buffer: Union[str, pathlib.Path, IO[~AnyStr]], sep=',', delimiter=None, header='infer', names=None, index_col=None,
#                usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None,
#                false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True,
#                na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
#                date_parser=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, 
#                decimal: str = '.', lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None,  encoding=None,
#                dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None)
dataset = read_csv(url, names=names)

#print(dataset.shape)  #print the shape of the dataset
#print(dataset.head(30))  #print the first 30 rows of data
#print(dataset.describe())  #dataset description
#print(dataset.groupby('class').size())  #see the categories of the flowers




#4.1 Univariate plots
## box and whisker plots 
#dataset.plot(kind='box', subplots=True, layout=(1,5), sharex=True, sharey=True)
#pyplot.show()

#dataset.plot(kind='line')
#pyplot.show()

##plot histogram via two methods
#dataset.plot(kind='hist')  #plot a histogram of four subplots in one big graph
#pyplot.show()

#dataset.hist()  #four separate histograms, one for each type
#pyplot.show()


##4.2 multivariate plots

## scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()




# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
#Split arrays or matrices into random train and test subsets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


#The general k-fold procedure is as follows:

# 1. Shuffle the dataset randomly.
# 2. Split the dataset into k groups
# 3. For each unique group:
#     - Take the group as a hold out or test data set
#     - Take the remaining groups as a training data set
#     - Fit a model on the training set and evaluate it on the test set
#     - Retain the evaluation score and discard the model
# 4. Summarize the skill of the model using the sample of model evaluation scores

#k=10: a value that has been found through experimentation to generally result in a model skill estimate with low bias a modest variance.


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# Choose an algorithm to make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))