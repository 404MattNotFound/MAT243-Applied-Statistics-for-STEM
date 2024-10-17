# coding: utf-8

# # Project Two: Hypothesis Testing

# ## Step 1: Data Preparation & the Assigned Team
# This step uploads the data set from a CSV file. It also selects the Assigned Team for this analysis. 

# In[1]:
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from IPython.display import display, HTML

nba_orig_df = pd.read_csv('nbaallelo.csv')
nba_orig_df = nba_orig_df[(nba_orig_df['lg_id']=='NBA') & (nba_orig_df['is_playoffs']==0)]
columns_to_keep = ['game_id','year_id','fran_id','pts','opp_pts','elo_n','opp_elo_n', 'game_location', 'game_result']
nba_orig_df = nba_orig_df[columns_to_keep]

# The dataframe for the assigned team is called assigned_team_df. 
# The assigned team is the Bulls from 1996-1998.
assigned_years_league_df = nba_orig_df[(nba_orig_df['year_id'].between(1996, 1998))]
assigned_team_df = assigned_years_league_df[(assigned_years_league_df['fran_id']=='Bulls')]
assigned_team_df = assigned_team_df.reset_index(drop=True)

display(HTML(assigned_team_df.head().to_html()))
print("printed only the first five observations...")
print("Number of rows in the dataset =", len(assigned_team_df))


# ## Step 2: Pick Your Team
# In this step, you will pick your team. The range of years that you will study for your team is 2013-2015.

# In[3]:

# Range of years: 2013-2015 (Note: The line below selects all teams within the three-year period 2013-2015)
your_years_leagues_df = nba_orig_df[(nba_orig_df['year_id'].between(2013, 2015))]

# The dataframe for your team is called your_team_df.
your_team_df = your_years_leagues_df[(your_years_leagues_df['fran_id']=='Celtics')]
your_team_df = your_team_df.reset_index(drop=True)

display(HTML(your_team_df.head().to_html()))
print("printed only the first five observations...")
print("Number of rows in the dataset =", len(your_team_df))


# ## Step 3: Hypothesis Test for the Population Mean (I)
# A relative skill level of 1340 represents a critically low skill level in the league. The management of your team has hypothesized that the average relative skill level of your team in the years 2013-2015 is greater than 1340. Test this claim using a 5% level of significance. 


# In[14]:

import scipy.stats as st

# Mean relative skill level of your team
mean_elo_your_team = your_team_df['elo_n'].mean()
print("Mean Relative Skill of your team in the years 2013 to 2015 =", round(mean_elo_your_team,2))

# Hypothesis Test
test_statistic, p_value = st.ttest_1samp(your_team_df['elo_n'], 1340)

print("Hypothesis Test for the Population Mean")
print("Test Statistic =", round(test_statistic,2)) 
print("P-value =", round(p_value,4)) 


#   

# ## Step 4: Hypothesis Test for the Population Mean (II)
# 
# A team averaging 106 points is likely to do very well during the regular season. The coach of your team has hypothesized that your team scored at an average of less than 106 points in the years 2013-2015. Test this claim at a 1% level of significance.  

# In[20]:

# Write your code in this code block section
import scipy.stats as st

# Calculate the mean points scored by your team
mean_pts = your_team_df['pts'].mean()

# Printing anyways to gather the most information
print(f"Mean points scored by the team: {mean_pts:.2f}")

# Identify the mean score under the null hypothesis
null_hypothesis_mean = 106

# Perform the t-test
t_stat, p_value = st.ttest_1samp(your_team_df['pts'], null_hypothesis_mean)

# Adjust for one-tailed test
p_value_one_tailed = p_value / 2  

# Print the test statistic and the P-value
print(f"Test statistic: {t_stat:.2f}")
print(f"P-value (one-tailed): {p_value_one_tailed:.4f}")

# ## Step 5: Hypothesis Test for the Population Proportion
# Suppose the management claims that the proportion of games that your team wins when scoring 102 or more points is 0.90. Test this claim using a 5% level of significance.

# In[22]:

from statsmodels.stats.proportion import proportions_ztest

your_team_gt_102_df = your_team_df[(your_team_df['pts'] > 102)]

# Number of games won when your team scores over 102 points
counts = (your_team_gt_102_df['game_result'] == 'W').sum()

# Total number of games when your team scores over 102 points
nobs = len(your_team_gt_102_df['game_result'])

p = counts*1.0/nobs
print("Proportion of games won by your team when scoring more than 102 points in the years 2013 to 2015 =", round(p,4))

# Hypothesis Test
test_statistic, p_value = proportions_ztest(counts, nobs, 0.90)

print("Hypothesis Test for the Population Proportion")
print("Test Statistic =", round(test_statistic,2)) 
print("P-value =", round(p_value,4))


# ## Step 6: Hypothesis Test for the Difference Between Two Population Means
# The management of your team wants to compare the team with the assigned team (the Bulls in 1996-1998). They claim that the skill level of your team in 2013-2015 is the same as the skill level of the Bulls in 1996 to 1998. In other words, the mean relative skill level of your team in 2013 to 2015 is the same as the mean relative skill level of the Bulls in 1996-1998. 

# In[23]:

import scipy.stats as st

mean_elo_n_project_team = assigned_team_df['elo_n'].mean()
print("Mean Relative Skill of the assigned team in the years 1996 to 1998 =", round(mean_elo_n_project_team,2))

mean_elo_n_your_team = your_team_df['elo_n'].mean()
print("Mean Relative Skill of your team in the years 2013 to 2015  =", round(mean_elo_n_your_team,2))

# Hypothesis Test
test_statistic, p_value = st.ttest_ind(assigned_team_df['elo_n'], your_team_df['elo_n'])

print("Hypothesis Test for the Difference Between Two Population Means")
print("Test Statistic =", round(test_statistic,2)) 
print("P-value =", round(p_value,4))