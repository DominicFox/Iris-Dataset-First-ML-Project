# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


# Load Dataset
file = "YOUR FILEPATH"
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file,names = columns)

# Split out the validation data set
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,y,test_size=0.20,random_state=1)

# Setup each model
models = []
models.append(('LogisticRegression', LogisticRegression(solver='liblinear')))
models.append(('KNearestNeighbours', KNeighborsClassifier()))
models.append(('SupportVectorMachine', SVC()))
models.append(('RidgeRegression', RidgeClassifier()))
models.append(('ClassificationAndRegressionTree', DecisionTreeClassifier()))
models.append(('GaussianNaiveBayes', GaussianNB()))
models.append(('MultinomialNaiveBayes', MultinomialNB()))
models.append(('MultilayerPerceptrons', MLPClassifier(max_iter=800)))
models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))

# Evaluate the accuracy of each model in turn
for currentName, currentModel in models:
	kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
	cv_results = cross_val_score(currentModel,X_train,Y_train,cv=kfold,scoring='accuracy')
	print('%s: %f (%f)' % (currentName, cv_results.mean(), cv_results.std()))















