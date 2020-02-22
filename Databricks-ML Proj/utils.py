import numpy as np
import seaborn as sns
from scipy import stats


def find_corr(df, y):
	"""Helper function to display correlation between features and target column
	Args:
		df: dataframe
		y (str): referenced column for correlation
	Return:
		None
	"""
	df_corr = df.corr()[y].sort_values(ascending=False)
	df_num_cols = df_corr.index.tolist()
	df_num = df[df_num_cols]
	sns.pairplot(df_num)


def detect_outlier_zscore(df):
	"""Helper function to detect outliers based on Z score
	Args:
		df: dataframe
	Return:
		outliers: tuple
	"""
	threshold = 3
	z = np.abs(stats.zscore(df))
	outliers = np.where(z > threshold)
	return outliers


def detect_outlier_iqr(df):
	"""Helper function to detect outliers based on IQR
	Args:
		df: dataframe
	Return:
		outliers: pandas dataframe
	"""
	Q1 = df.quantile(0.25)
	Q3 = df.quantile(0.75)
	IQR = Q3 - Q1
	outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
	return outliers
