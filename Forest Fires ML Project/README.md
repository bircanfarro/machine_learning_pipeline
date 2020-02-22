
Databricks ML Proj;
    1. data
    2. demo
    3. models_factory
    4. utils.py 
    5. pipeline.py
    6. main.py

The purpose of this project is to showcase a Machine Learning pipeline rather than creating the best models.

1. `data` has `forest_fires.csv`

2. `demo` has jupyter notebook workflow with more comprehensive content.

3. `models_factory` has three factories that create basic models. All the factories have the same input and return `predict()` functions. The architecture allows any type of model to be created and trained.

4. `utils` has some funtions to see the correlation between columns and outliers in data. It can be improved by adding more utility functions.

5. `pipeline` has a `Pipeline` class that allows model factories to be added to it in a generic way. Each factory has a `weight` associatd with it and must return a `predict()` function. The `Pipeline` `train()` method creates and trains a model from each factory. All models are created at the same time. The package `predict()` function predicts values based on the current models. All predicted values are averaged based on the `weight` assigned to each factory.

6. `main` is where the pipeline is being demonstrated. In order to keep the demo simple, the day and month columns are dropped and outliers are not removed.

To execute the project, download the file and go to the main directory. In command line run python main.py.

output:
sample data
    X  Y month  day  FFMC   DMC     DC  ISI  temp  RH  wind  rain  area
0  7  5   mar  fri  86.2  26.2   94.3  5.1   8.2  51   6.7   0.0   0.0
1  7  4   oct  tue  90.6  35.4  669.1  6.7  18.0  33   0.9   0.0   0.0
2  7  4   oct  sat  90.6  43.7  686.9  6.7  14.6  33   1.3   0.0   0.0
3  8  6   mar  fri  91.7  33.3   77.5  9.0   8.3  97   4.0   0.2   0.0
4  8  6   mar  sun  89.3  51.3  102.2  9.6  11.4  99   1.8   0.0   0.0
---------Train Results---------
pipe train mae   21.6866779198
pipe train rmse  117.839896591
---------Test Results----------
pipe test mae    11.2797947148
pipe test rmse   27.2315903404


