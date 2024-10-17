# ## Step 1: Data Preparation

# In[1]:
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# dataframe for this project
nba_wins_df = pd.read_csv('nba_wins_data.csv')

display(HTML(nba_wins_df.head().to_html()))
print("printed only the first five observations...")
print("Number of rows in the dataset =", len(nba_wins_df))

# ## Step 2: Scatterplot and Correlation for the Total Number of Wins and Average Relative Skill
# Your coach expects teams to win more games in a regular season if they have a higher average relative skill compared to their opponents. This is because the chances of winning are higher if a team can maintain high average relative skill. Therefore, it is expected that the total number of wins and the average relative skill are correlated. Calculate the Pearson correlation coefficient and its P-value.

# In[5]:
import scipy.stats as st

plt.plot(nba_wins_df['avg_elo_n'], nba_wins_df['total_wins'], 'o')
plt.title('Total Number of Wins by Average Relative Skill', fontsize=20)
plt.xlabel('Average Relative Skill')
plt.ylabel('Total Number of Wins')
plt.show()

correlation_coefficient, p_value = st.pearsonr(nba_wins_df['avg_elo_n'], nba_wins_df['total_wins'])

print("Correlation between Average Relative Skill and the Total Number of Wins ")
print("Pearson Correlation Coefficient =",  round(correlation_coefficient,4))
print("P-value =", round(p_value,4))


## Step 3: Simple Linear Regression: Predicting the Total Number of Wins using Average Relative Skill

# In[6]:
import statsmodels.formula.api as smf

# Simple Linear Regression
model1 = smf.ols('total_wins ~ avg_pts', nba_wins_df).fit()
print(model1.summary())

# ## Step 4: Scatterplot and Correlation for the Total Number of Wins and Average Points Scored
# Your coach expects teams to win more games in a regular season if they score more points on average during the season. This is because the chances of winning are higher if a team maintains high average points scored. Therefore, it is expected that the total number of wins and the average points scored are correlated. Calculate the Pearson correlation coefficient and its P-value.

# In[10]:
import scipy.stats as st

plt.plot(nba_wins_df['avg_pts'], nba_wins_df['total_wins'], 'o')
plt.title('Total Number of Wins by Average Points Scored', fontsize=20)
plt.xlabel('Average Points Scored')
plt.ylabel('Total Number of Wins')
plt.show()

correlation_coefficient, p_value = st.pearsonr(nba_wins_df['avg_pts'], nba_wins_df['total_wins'])

print("Correlation between Average Points Scored and the Total Number of Wins ")
print("Pearson Correlation Coefficient =",  round(correlation_coefficient,4))
print("P-value =", round(p_value,4))

# ## Step 5: Multiple Regression: Predicting the Total Number of Wins using Average Points Scored and Average Relative Skill
# 
# Instead of presenting a simple linear regression model to the coach, you can suggest a multiple regression model with the total number of wins as the response variable and the average points scored and the average relative skill as predictor variables. This regression model will help your coach predict how many games your team might win in a regular season based on metrics like the average points scored and average relative skill. This model is more practical because you expect more than one performance metric to determine the total number of wins in a regular season. 


# In[11]:
import statsmodels.formula.api as smf

# Multiple Regression
model2 = smf.ols('total_wins ~ avg_pts + avg_elo_n', nba_wins_df).fit()
print(model2.summary())

# ## Step 6: Multiple Regression: Predicting the Total Number of Wins using Average Points Scored, Average Relative Skill, Average Points Differential and Average Relative Skill Differential
# 
# The coach also wants you to consider the average points differential and average relative skill differential as predictor variables in the multiple regression model. Create a multiple regression model with the total number of wins as the response variable, and average points scored, average relative skill, average points differential and average relative skill differential as predictor variables. This regression model will help your coach predict how many games your team might win in a regular season based on metrics like the average score, average relative skill, average points differential and average relative skill differential between the team and their opponents. 

# In[13]:
import statsmodels.formula.api as smf

# Multiple Regression
model3 = smf.ols('total_wins ~ avg_pts + avg_elo_n + avg_pts_differential', nba_wins_df).fit()

print(model3.summary())