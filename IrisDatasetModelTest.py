# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load Dataset
file = "YOUR FILEPATH"
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file,names = columns)

# Split out the validation data set
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,y,test_size=0.20,random_state=1)

# Test the LDA model on the validation set
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print("Predictions on validation set: \n", predictions, "\n")

# Evaluate the predictions
print("Accuracy Score: %f" % (accuracy_score(Y_validation, predictions)), "\n")
print("Confusion Matrix: \n", confusion_matrix(Y_validation, predictions), "\n")
print("Classifcation Report: \n", classification_report(Y_validation, predictions))












