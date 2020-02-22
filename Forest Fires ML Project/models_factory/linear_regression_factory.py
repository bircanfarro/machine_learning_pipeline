from sklearn.linear_model import LinearRegression

def factory(X, y):
	lm = LinearRegression()
	lm.fit(X, y)
	return lm.predict

#TODO: update the function with log transform
####sudo code
# def factory(X, y):
# 	X_pruned = prune_outliers(X)
# 	# test with fit
# 	lnr = log_transform()
# 	lnr.fit(X_pruned)
# 	X_scaled = lnr.scale(X_pruned)
# 	fitted_lm = LinearRegression()
# 	fitted_lm.fit(X_scaled, y)
# 	rmse_w_fit = fitted_lm.predict()

# 	lm = LinearRegression()
# 	lm.fit(X_pruned)
# 	rmse_wo_fit = lm.predict(X_pruned)

# 	use_fit = rmse_w_fit < rmse_wo_fit

# 	def predict(X_test):
# 		if use_fit:
# 			X_test_scaled = lnr.scale(X_test)
# 			return fitted_lm.predict(X_test_scaled)
# 		lm.predict(X_test)

# 	return predict

