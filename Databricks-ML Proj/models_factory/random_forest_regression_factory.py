from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def factory(X, y):
	"""Random Forest Model with pre-hyperparameter tuned"""
	gsc = GridSearchCV(
		estimator=RandomForestRegressor(),
		param_grid={
				'max_depth': range(3,7),
				'n_estimators': (10, 50, 100, 1000)},
		cv=5,
		verbose=0,
		n_jobs=-1
	)

	grid_result = gsc.fit(X, y)
	best_params = grid_result.best_params_
	rfr = RandomForestRegressor(
		max_depth=best_params["max_depth"],
		n_estimators=best_params["n_estimators"],
		random_state=False,
		verbose=False)
	rfr.fit(X, y)

	return rfr.predict



# Derived from: https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb
