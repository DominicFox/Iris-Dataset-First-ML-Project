# Load Libraries
from pandas import read_csv
from seaborn import pairplot
from matplotlib import pyplot

# Load Dataset
file = "YOUR FILEPATH"
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file,names = columns)

# Print Information about Data
print(dataset.shape)
print(dataset.head(30))
print(dataset.describe())
print(dataset.groupby('class').size())

# Box and Whisker plot
dataset.plot(kind='box')
pyplot.show()

# Histogram
dataset.hist(bins = 20)
pyplot.show()

# Scatter Matrix / Pairplot
pairplot(dataset, hue = 'class')
pyplot.show()









