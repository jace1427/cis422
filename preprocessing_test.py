"""
unittests for the data manipulation functions in processing.py

unittest does not have a built in function for comparing pandas DataFrame's, but it does for lists.
The test cases obtain a list from the data column of the DataFrame and compares these lists.

TO RUN: simply execute the following command in the terminal "python3 preprocessing_test.py"

Author: Riley Matthews

"""

# ------------------------ IMPORTS ------------------------ #

import unittest
import preprocessing as pp
import fileIO

# ----------------------- Test Cases ---------------------- #


class Testpreprocessing(unittest.TestCase):

	def setUp(self):
		#print('setUp')
		self.data = fileIO.csv_to_dataframe("TestData/test_data.csv")
		self.data_missing = fileIO.csv_to_dataframe("TestData/test_missing_data.csv")
		self.scaling_data = fileIO.csv_to_dataframe("TestData/test_scaling.csv")
		self.difference_data = fileIO.csv_to_dataframe("TestData/test_difference.csv")
		self.logarithm_data = fileIO.csv_to_dataframe("TestData/test_logarithm.csv")
		self.cubic_data = fileIO.csv_to_dataframe("TestData/test_cubic_root.csv")
		self.standardize_data = fileIO.csv_to_dataframe("TestData/test_standardize.csv")
		self.missing_times = fileIO.read_from_file_no_check("TestData/test_assign_time.csv")


	def test_denoise(self):
		"""denoise was tested manually by comparing plots of graphs."""
		print('test_denoise')
		pass


	def test_impute_missing_data(self):
		print('test_impute_missing_data')

		# test backfill
		self.assertEqual(pp.impute_missing_data(self.data_missing, 'backfill').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_bfill_result.csv").iloc[:, 1].to_list())

		# test forward fill
		self.assertEqual(pp.impute_missing_data(self.data_missing, 'forwardfill').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_ffill_result.csv").iloc[:, 1].to_list())

		# test mean
		self.assertEqual(pp.impute_missing_data(self.data_missing, 'mean').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_mean_result.csv").iloc[:, 1].to_list())

		# test median
		self.assertEqual(pp.impute_missing_data(self.data_missing, 'median').iloc[:, 1].to_list(), 
			fileIO.csv_to_dataframe("TestData/test_impute_missing_data_median_result.csv").iloc[:, 1].to_list())


	def test_impute_outliers(self):
		print('test_impute_outliers')
		
		self.assertEqual(pp.impute_outliers(self.data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_impute_outliers_result.csv").iloc[:, 1].to_list())


	def test_longest_continuous_run(self):
		print('test_longest_continuous_run')
		
		self.assertEqual(pp.longest_continuous_run(self.data_missing).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_longest_run_result.csv").iloc[:, 1].to_list())


	def test_clip(self):
		print('test_clip')
		
		# test a clip at the start
		self.assertEqual(pp.clip(self.data, 0, 4).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_1_result.csv").iloc[:, 1].to_list())

		# test a clip in the middle
		self.assertEqual(pp.clip(self.data, 4, 12).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_2_result.csv").iloc[:, 1].to_list())

		# test a clip at the end
		self.assertEqual(pp.clip(self.data, 14, 19).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_clip_3_result.csv").iloc[:, 1].to_list())

		# test invalid starting index
		self.assertRaises(IndexError, pp.clip, self.data, -1, 5)
		self.assertRaises(IndexError, pp.clip, self.data, 6, 4)
		self.assertRaises(IndexError, pp.clip, self.data, 20, 17)

		# test invalid final index
		self.assertRaises(IndexError, pp.clip, self.data, 7, 20)
		self.assertRaises(IndexError, pp.clip, self.data, 2, -5)
		self.assertRaises(IndexError, pp.clip, self.data, 10, 6)


	def test_assign_time(self):
		print('test_assign_time')

		self.assertEqual(pp.assign_time(self.missing_times, '1/1/2021', 5).iloc[:, 0].to_list(),
			fileIO.csv_to_dataframe("TestData/test_assign_time_result.csv").iloc[:, 0].to_list())
		

	def test_difference(self):
		print('test_difference')

		self.assertEqual(pp.difference(self.difference_data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_difference_result.csv").iloc[:, 1].to_list())


	def test_scaling(self):
		print('test_scaling')

		self.assertEqual(pp.scaling(self.scaling_data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_scaling_result.csv").iloc[:, 1].to_list())


	def test_standardize(self):
		print('test_standardize')
		
		self.assertEqual(pp.standardize(self.standardize_data).iloc[5, 1], 0)


	def test_logarithm(self):
		print('test_logarithm')
		
		self.assertEqual(pp.logarithm(self.logarithm_data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_logarithm_result.csv").iloc[:, 1].to_list())


	def test_cubic_root(self):
		print('test_cubic_root')
		
		self.assertEqual(pp.cubic_root(self.cubic_data).iloc[:, 1].to_list(),
			fileIO.csv_to_dataframe("TestData/test_cubic_root_result.csv").iloc[:, 1].to_list())


if __name__ == '__main__':
	unittest.main()