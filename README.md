# Keras-GridSearchCV
Workaround for using GridsearchCV with kerasWrapper (KerasClassifier and KerasRegressor) + tensorflow without getting Out of Memory errors. This is a solution for problems like [This](https://stackoverflow.com/questions/42047497/keras-out-of-memory-when-doing-hyper-parameter-grid-search), using a conveniently simple interface for defining the grid search and finding the best parameters (sklearn [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)). The main difference with sklearn implementation is that the keras backend session is passed as an argument to the fit method of the GridSearchCV class, and also, maintaining all the original parameters.

**Warning:** This will not avoid OOM errors if your model is too big for your GPU(s)

**Requirements:**
- Scikit-learn == 0.19.1
- Tensorflow-gpu <= 1.8.0
- Keras <= 2.2.2

-----------------------

## Some notes

- Currently, the main solution (kerasGridSearch) does not implement parallelization provided in GridSearchCV (use it with fixed n_jobs=1). Using n_jobs greater than 1 can produce some problems with [tensorflow hanging](https://stackoverflow.com/questions/47527915/keras-scikit-learn-wrapper-appears-to-hang-when-gridsearchcv-with-n-jobs-1).  To overcome this, there is an implementation (kerasGridSearch2) which instead of using the wrapper, it recieves the model creation method and a type of estimator ("Classifier" or "Regressor"). For more information about, please see the [example number 2](https://github.com/capkuro/Keras-GridSearchCV/blob/master/example/GridsearchTest2.py)

- little benchmark using kerasGridSearch2 and parallelization (with Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz and NVIDIA 1080ti 11GB, driver version 384.130 and cudatoolkit from anaconda):
	-  `[Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed:  2.2min finished`
	-  `[Parallel(n_jobs=2)]: Done  54 out of  54 | elapsed:  1.2min finished`
	-  `[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:   49.7s finished`
	-  `[Parallel(n_jobs=8)]: Done  54 out of  54 | elapsed:   39.9s finished`


- With multiple GPUS, you can direct training on a specific GPU with the `CUDA_VISIBLE_DEVICES` flag such as:

```
$ CUDA_VISIBLE_DEVICES=0 GridsearchTest.py
```

- To avoid "I" log messages for each new session created, you could start with `TF_CPP_MIN_LOG_LEVEL=2` flag  and adding `tf.logging.set_verbosity(tf.logging.ERROR)` inside the code. To execute this on a terminal you can use `export` before executing your code.

```
$ export TF_CPP_MIN_LOG_LEVEL=2
$ python GridsearchTest.py
```

------------------------

## Acknowledgments

- Jason Brownlee, Ph.D from [Machine Learning Mastery](https://machinelearningmastery.com/), for the [example](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/) using GridSearchCV and the KerasClassifier wrapper
- Scikit-learn Team
- Github user [nikhilweee](https://github.com/nikhilweee) for the [solution](https://github.com/tensorflow/tensorflow/issues/566#issuecomment-296193947) in avoiding "I" log messages
- Github user [mmry](https://github.com/mrry) for [hintting](https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405) the solution of the multiprocess tensorflow hang problem back in 2016
