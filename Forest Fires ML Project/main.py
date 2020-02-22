import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pipeline
import models_factory.linear_regression_factory as lrm
import models_factory.random_forest_regression_factory as rfrm
import models_factory.support_vector_regression_factory as svrm

# Read and split data in train and test
df = pd.read_csv('data/forest_fires.csv')
print('sample data\n', df.head())
train = df.sample(frac=0.8, random_state=10) #random state is a seed value
holdout = df.drop(train.index)


# TRAINING

# Train models with the Pipeline class
X = train.drop(['area'], axis=1)
X = X.drop(['day', 'month'], axis=1)
y = train['area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 0)

demo_pipe = pipeline.Pipeline()
# Add different models to the factory
demo_pipe.add_model_factory('lr', 'Linear Regression', weight=1, factory=lrm.factory)
demo_pipe.add_model_factory('rfr', 'Random Forest Regression', weight=1, factory=rfrm.factory)
# SVR results best mae, so it's weighted higher than the other models
demo_pipe.add_model_factory('svr', 'Support Vector Regresion', weight=3, factory=svrm.factory)
# All model are added to the pipeline and trained at the same time
demo_pipe_model = demo_pipe.train(X_train, y_train)

# Train/Test models with the Prediction function
pipe_test_pred = pipeline.predict(demo_pipe_model, X_test)

# Evaluate model performance by mae and rmse
mae = metrics.mean_absolute_error(y_test, pipe_test_pred)
rmse = metrics.mean_squared_error(y_test, pipe_test_pred) ** 0.5
print('---------Train Results---------')
print('pipe train mae\t', mae)
print('pipe train rmse\t', rmse)


# TESTING

# Test the models with test(holdout) data
X_holdout = holdout.drop(['area'], axis=1)
X_holdout = X_holdout.drop(['day', 'month'], axis=1)
y_holdout = holdout['area']

pipe_holdout_pred = pipeline.predict(demo_pipe_model, X_holdout)

# Evaluate model performance by mae and rmse
mae = metrics.mean_absolute_error(y_holdout, pipe_holdout_pred)
rmse = metrics.mean_squared_error(y_holdout, pipe_holdout_pred) ** 0.5
print('---------Test Results----------')
print('pipe test mae\t', mae)
print('pipe test rmse\t', rmse)


