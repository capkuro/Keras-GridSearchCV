# Keras-GridSearchCV
Workaround for using GridsearchCV with kerasWrapper (KerasClassifier and KerasRegressor) + tensorflow without getting Out of Memory errors. This is a solution for problems like [This](https://stackoverflow.com/questions/42047497/keras-out-of-memory-when-doing-hyper-parameter-grid-search), using a conveniently simple interface for defining the grid search and finding the best parameters (sklearn [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)). The main difference with sklearn implementation is that the keras backend session is passed as an argument to the fit method of the GridSearchCV class, and also, maintaining all the original parameters.

**Warning:** This will not avoid OOM errors if your model is too big for your GPU(s)

**Requirements:**
- Scikit-learn == 0.19.1
- Tensorflow-gpu <= 1.8.0
- Keras <= 2.2.2

-----------------------

## Some notes

Currently, this solution does not implement parallelization provided in GridSearchCV (use it with fixed n_jobs=1). Using n_jobs greater than 1 can produce some problems with tensorflow hanging [link](https://stackoverflow.com/questions/47527915/keras-scikit-learn-wrapper-appears-to-hang-when-gridsearchcv-with-n-jobs-1)

------------------------

## Acknowledgments

- Jason Brownlee, Ph.D from [Machine Learning Mastery](https://machinelearningmastery.com/)++, for the ++[example](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/) using GridSearchCV and the KerasClassifier wrapper
- Scikit-learn Team