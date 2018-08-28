import sys
sys.path.append('../') #Included to do the import of kerasGridSearchCV
import numpy
from gridsearch.kerasGridSearch import kerasGridSearchCV2  # Import Gridsearch

# The example code comes mainly from
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# Function to create model, required for KerasClassifier

# Code from Machine learning mastery
def create_model():
	# create model
	from keras.models import Sequential
	from keras.layers import Dense
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
# Instead of GridsearchCV, kerasGridSearchCV2 which implements a major change in the fit method.
# This class instead of recieving a model, we pass the build function to kerasGridSearchCV2
# also we have to remove any import on the main code of keras and tensorflow, to avoid a session for creating
# With this, we can paralellize the grid search if we have enough GPU memory for each parallel process
grid = kerasGridSearchCV2(estimator=create_model, param_grid=param_grid, n_jobs=2, cv=3,verbose=1)
grid_result = grid.fit(X, Y, type = "Classifier") #We pass the type = "Classifier" or "Regressor" in the type parameter of the fit method
# Code from machine learning mastery
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))