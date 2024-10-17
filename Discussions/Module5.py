# Module Five Discussion: Correlation and Simple Linear Regression

import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# read data from mtcars.csv data set.
cars_df_orig = pd.read_csv("https://s3-us-west-2.amazonaws.com/data-analytics.zybooks.com/mtcars.csv")

# randomly pick 30 observations without replacement from mtcars dataset to make the data unique to you.
cars_df = cars_df_orig.sample(n=30, replace=False)

# print only the first five observations in the data set.
print("\nCars data frame (showing only the first five observations)")
display(HTML(cars_df.head().to_html()))

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

# create scatterplot of variables mpg against wt.
plt.plot(cars_df["wt"], cars_df["mpg"], 'o', color='red')

# set a title for the plot, x-axis, and y-axis.
plt.title('MPG against Weight')
plt.xlabel('Weight (1000s lbs)')
plt.ylabel('MPG')

# show the plot.
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

# create correlation matrix for mpg and wt. 
# the correlation coefficient between mpg and wt is contained in the cell for mpg row and wt column (or wt row and mpg column) 
mpg_wt_corr = cars_df[['mpg','wt']].corr()
print(mpg_wt_corr)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -

# create the simple linear regression model with mpg as the response variable and weight as the predictor variable
model = ols('mpg ~ wt', data=cars_df).fit()

#print the model summary
print(model.summary())