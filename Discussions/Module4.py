# Module Four Discussion: Hypothesis Testing for the Difference in Two Population Proportions

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# create 50 randomly chosen values from a normal distribution. (arbitrarily using mean=2.48 and standard deviation=0.500) 
diameters_sample1 = np.random.normal(2.48,0.500,50)

# convert the array into a dataframe with the column name "diameters" using pandas library
diameters_sample1_df = pd.DataFrame(diameters_sample1, columns=['diameters'])
diameters_sample1_df = diameters_sample1_df.round(2)

# create 50 randomly chosen values from a normal distribution. (arbitrarily using mean=2.50 and standard deviation=0.750) 
diameters_sample2 = np.random.normal(2.50,0.750,50)

# convert the array into a dataframe with the column name "diameters" using pandas library
diameters_sample2_df = pd.DataFrame(diameters_sample2, columns=['diameters'])
diameters_sample2_df = diameters_sample2_df.round(2)

# print the dataframe to see the first 5 observations (note that the index of dataframe starts at 0)
print("Diameters data frame of the first sample (showing only the first five observations)")
print(diameters_sample1_df.head())
print()
print("Diameters data frame of the second sample (showing only the first five observations)")
print(diameters_sample2_df.head())

# number of observations in the first sample with diameter values less than 2.20. 
count1 = len(diameters_sample1_df[diameters_sample1_df['diameters']<2.20])

# number of observations in the second sample with diameter values less than 2.20. 
count2 = len(diameters_sample2_df[diameters_sample2_df['diameters']<2.20])

# counts Python list
counts = [count1, count2]


# number of observations in the first sample
n1 = len(diameters_sample1_df)

# number of observations in the second sample
n2 = len(diameters_sample2_df)

# n Python list
n = [n1, n2]

# perform the hypothesis test. output is a Python tuple that contains test_statistic and the two-sided P_value.
test_statistic, p_value = proportions_ztest(counts, n)

print()
print("test-statistic =", round(test_statistic,2))
print("two tailed p-value =", round(p_value,4))