from sklearn.svm import SVR

def factory(X, y):
	"""Base SVR Model"""
	svr = SVR(kernel='rbf')
	svr.fit(X, y)
	return svr.predict
