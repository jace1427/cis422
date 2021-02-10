"""
unittests for the functions in preprocessing.py

unittest does not have a built in function for comparing pandas DataFrame's but it does for lists.
The test cases obtain a list from the data column of the DataFrame and compares these lists.

TO RUN: simply execute the following command in the terminal "python3 preprocessing_test.py"

Author: Riley Matthews

"""

# ------------------------ IMPORTS ------------------------ #

import unittest
import preprocessing
import fileIO

# ----------------------- Test Cases ---------------------- #


class Testpreprocessing(unittest.TestCase):

	def setUp(self):
		#print('setUp')
		self.data = fileIO.csv_to_dataframe("TestData/test_data.csv")
		self.data_missing = fileIO.csv_to_dataframe("TestData/test_missing_data.csv")

	def test_denoise(self):
		print('test_denoise')
		pass

	def test_impute_missing_data(self):
		print('test_impute_missing_data')

		# test backfill
		self.assertEqual(preprocessing.impute_missing_data(self.data_missing, 'backfill').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_bfill_result.csv").iloc[:, 1].to_list())

		# test forward fill
		self.assertEqual(preprocessing.impute_missing_data(self.data_missing, 'forwardfill').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_ffill_result.csv").iloc[:, 1].to_list())

		# test mean
		self.assertEqual(preprocessing.impute_missing_data(self.data_missing, 'mean').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_mean_result.csv").iloc[:, 1].to_list())

		# test median
		self.assertEqual(preprocessing.impute_missing_data(self.data_missing, 'median').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_median_result.csv").iloc[:, 1].to_list())

	def test_impute_outliers(self):
		print('test_impute_outliers')
		
		self.assertEqual(preprocessing.impute_outliers(self.data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_impute_outliers_result.csv").iloc[:, 1].to_list())

	def test_longest_continuous_run(self):
		print('test_longest_continuous_run')
		
		self.assertEqual(preprocessing.longest_continuous_run(self.data_missing).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_longest_run_result.csv").iloc[:, 1].to_list())

	def test_clip(self):
		print('test_clip')
		
		# test a clip at the start
		self.assertEqual(preprocessing.clip(self.data, 0, 4).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_1_result.csv").iloc[:, 1].to_list())

		# test a clip in the middle
		self.assertEqual(preprocessing.clip(self.data, 4, 12).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_2_result.csv").iloc[:, 1].to_list())

		# test a clip at the end
		self.assertEqual(preprocessing.clip(self.data, 14, 19).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_3_result.csv").iloc[:, 1].to_list())

		# test a clip with an invalid starting date

		# test a clip with an invalid final date

	def test_assign_time(self):
		print('test_assign_time')
		pass

	def test_difference(self):
		print('test_difference')
		pass

	def test_scaling(self):
		print('test_scaling')
		pass

	def test_standardize(self):
		print('test_standardize')
		pass

	def test_logarithm(self):
		print('test_logarithm')
		pass

	def test_cubic_root(self):
		print('test_cubic_root')
		pass

if __name__ == '__main__':
	unittest.main()