#!/usr/bin/env python
# coding: utf-8

# # import necessary packages

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
import statsmodels.api as sm


# # import dataset

# In[3]:


data=pd.read_excel("/home/unni/Documents/ml/dataset/auto-mpg.xlsx",na_values=["?"])


# In[5]:


# top 5 rows
data.head()


# # exploratory data analysis

# In[6]:


# column data types, non-null count etc
data.info()


# In[8]:


# 5 number summary
data.describe()


# In[9]:


# summary for object datatypes
data.describe(include="O")


# In[10]:


# relationship between input features
correlation=data.corr(numeric_only=True)
correlation


# ## Data Cleaning

# In[11]:


# Handle Nulls..if there are less number of null values , remove those...else use other methods like data imputation 
# here , remove rows having null value
data.dropna(axis='index',inplace=True)
data.info()


# ## Define Independent (x) and Dependent variable (y)

# In[13]:


# independent variable..take only one feature initially
x=data.iloc[:,0]
print("-----",data.columns[0],"-----\n",x[:6])
print(x.shape)

# dependent variable / target variable
y=data.iloc[:,3]
print("\n-----",data.columns[3],"-----\n",y[:6])
print(y.shape)


# # Correlation Coefficients
# (Pearson, Kendall etc)

# - correlation coeff's lies between -1 (negative correlation) and 1 (strong/positive correlation). We can ignore the features whose correlation coeff = 0 (no correlation)
# - we can plot the graph and confirm

# In[14]:


# pearson correlation coeff
coeff_p,unk=pearsonr(x,y)
print("pearson correlation coeff =",coeff_p)


# In[15]:


# spearman's rank correlation
coeff_s,unk=spearmanr(x,y)
print("spearman's rank correlation coeff =",coeff_s)


# In[16]:


# plot graph to visualize...what can you infer from graph ??
plt.scatter(x,y)
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[3])
plt.show()


# ## what can you infer from above graph ??

# ## Reshape to make it training compatible
# <b>X</b> to be in form <b>(n_samples,n_features)</b> ; <b>Y</b> to be <b>(nsamples,)</b><br> 

# In[17]:


# DO THIS ONLY IF YOU HAVE ONE INPUT FEATURE

x=np.array(x)
print(x)

x=x.reshape(-1, 1)
print(x)


# In[18]:


print(x.shape)
print(y.shape)


# # Divide into Train and Test set
# #### random_state -> default=None ; Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[20]:


print("x_train.count()=",len(x_train)) # x_train.count()
print("y_train.count()=",y_train.count())
print("x_test.count()=",len(x_test)) # x_test.count()
print("y_test.count()=",y_test.count())


# # Train model

# In[21]:


# initilize model
# If fit_intercept=False, no intercept will be used in 
# calculations (i.e. data is expected to be centered/pass through origin).

lr=LinearRegression()


# In[22]:


# fit model
lr.fit(x_train,y_train)


# In[23]:


# parameters from model

# intercept
print("Intercept ->",lr.intercept_)

# coeff
print("coefficient ->",lr.coef_)


# # Performance on TRAIN set

# In[24]:


# predict model on TRAIN set
y_train_pred=lr.predict(x_train)


# In[26]:


# Root Mean Square Error
mse=mean_squared_error(y_train,y_train_pred)
rmse=np.sqrt(mse)
print("RMSE (Train) ->", rmse)


# In[27]:


# mean_absolute_error
mae=mean_absolute_error(y_train,y_train_pred)
print("Mean Absolute Error (Train) ->", mae)


# In[25]:


# actual observations
plt.scatter(x_train,y_train) # single feature
#plt.scatter(x_train.iloc[:,0],y_train) # multiple feaures

# predicted values on TRAIN set
plt.plot(x_train,y_train_pred) # single feature
#plt.plot(x_train.iloc[:,0],y_train_pred,label="linear") # multiple feaures..for plot take only one

plt.title("Regression Line - Train set")
plt.xlabel(data.columns[0])
plt.legend()
plt.ylabel("Actual Vs Predicted Values (TRAIN SET) ->")

plt.show()


# # Performance on TEST set

# In[28]:


# predict model on TEST set
y_pred=lr.predict(x_test)


# In[30]:


# Root Mean Square Error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("RMSE (Test)->", rmse)


# In[31]:


# mean_absolute_error
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error (Test)->", mae)


# In[32]:


# actual observations
plt.scatter(x_test,y_test) # single-feature
#plt.scatter(x_test.iloc[:,0],y_test) # multiple features

# predicted values
plt.plot(x_test,y_pred) # single-feature
#plt.plot(x_test.iloc[:,0],y_pred)  # multiple features


plt.title("Regression Line")
plt.xlabel(data.columns[0])
plt.ylabel("Actual Vs Predicted Values ->")
plt.legend()

plt.show()


# # Predict a new data point

# In[34]:


# mpg = 9.5
print(lr.predict([[9.5]]))


# # Test Goodness of fit 

# ### R squared i:e coeff of determination

# - is a measure of variability in output variable (predicted) explained by input variable (predictors) (i:e <b>it evaluates how closely y values scatter around your regression line</b>, the closer they are to your regression line the better)
# - takes a value between <b>0 (poor fit) and 1 (good fit)</b>
# - shouldnot be taken as sole criteria to judge that linear model is adequate
# - <b>comes with an inherent problem</b> – additional input variables will make the R-squared stay the same or increase (this is due to how the R-squared is calculated mathematically). Therefore, even if the additional input variables show no relationship with the output variables, the R-squared will increase

# In[35]:


# R Squared value
r_squared=r2_score(y_test,y_pred)
print("R squared i:e coeff of determination ->",r_squared)


# ### Adjusted R squared
# - The <b> ADJUSTED R-squared </b> is a modified version of R-squared that adjusts for predictors that are not significant in a regression model.
# - The adjusted R-squared <b>looks at whether additional input variables are contributing to the model.</b>
# - An indicator of whether adding additional predictors improve a regression model or not
# - A <b>lower adjusted R-squared</b> indicates that the additional input variables are not adding value to the model.
# - A <b>higher adjusted R-squared</b> indicates that the additional input variables are adding value to the model.

# In[36]:


# no. of samples
n=len(y_test)
# no. of parameters/independent variables
p=x_test.shape[1]

print("n=",n,", p=",p)

adj_rsquared=1-((1-r_squared)*(n-1)/(n-p-1))
print("Adjusted R squared ->",adj_rsquared)


# ## Add additonal input features and see the effect on R-squared and Adj R squared

# #### Additional feature

# In[37]:


data=pd.read_excel("/home/unni/Documents/ml/dataset/auto-mpg.xlsx",na_values=["?"])
#print(data.iloc[:,[0,3]])
data.head()


# In[38]:


data.dropna(axis='index',inplace=True)
data.info()


# In[40]:


# independent variable
x=data.iloc[:,[0,1]] # mpg and cylinders

print(x.shape)
print(x[:6])

# dependent variable / target variable
y=data.iloc[:,3]
print(y[:6])
print(y.shape)


# In[41]:


# divide into train and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print("x_train.count()=",len(x_train)) # x_train.count()
print("y_train.count()=",y_train.count())
print("x_test.count()=",len(x_test)) # x_test.count()
print("y_test.count()=",y_test.count())


# In[42]:


# initilize model
lr=LinearRegression()

lr.fit(x_train,y_train)

# intercept
print("Interept ->",lr.intercept_)
# coeff
print("coefficient ->",lr.coef_)


# In[43]:


# predict model on test set
y_pred=lr.predict(x_test)


# In[44]:


# Root Mean Square Error

mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("RMSE ->", rmse)


# In[45]:


# R Squared
r_squared=r2_score(y_test,y_pred)
print("R squared i:e coeff of determination ->",r_squared)


# In[46]:


# Adjusted R Squared

# no. of samples
n=len(y_test)
# no. of parameters/independent variables
p=x_test.shape[1]

print("n=",n,", p=",p)

adj_rsquared=1-((1-r_squared)*(n-1)/(n-p-1))
print("Adjusted R squared ->",adj_rsquared)


# ### What do you understand from above ?? Is adding additional feature adding value ???

# # Hypothesis Tests
# As per <b>NULL Hypothesis</b> , Reduced model (y = b0) is sufficient and not y = b0 + b1x1 + b2x2...+bnxn  i:e Input features (x1,x2..xn) have no effect on Dependent/Target variable (y)...i:e b1 (beta 1), b2..bn= 0 ; Only b0 (beta 0, Intercept/bias) sufficient to explain model
# 
# We reject the Null Hypothesis, if p-value ("P>|t|" column) <0.05. So variables whose p-value < 0.05 are significant. Reject other variables whose p-value>0.05

# In[48]:


data=pd.read_excel("/home/unni/Documents/ml/dataset/auto-mpg.xlsx",na_values=["?"])
data.head()

data.dropna(axis='index',inplace=True)
data.info()


# In[49]:


# independent variable..take all
x=data.iloc[:,[0,1,2,4,5,6,7]]

print(x.shape)
print(x[:6])

# dependent variable / target variable
y=data.iloc[:,3]
print(y[:6])
print(y.shape)


# In[50]:


# divide into train and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print("x_train.count()=",len(x_train)) # x_train.count()
print("y_train.count()=",y_train.count())
print("x_test.count()=",len(x_test)) # x_test.count()
print("y_test.count()=",y_test.count())


# In[51]:


x_train_lm=sm.add_constant(x_train)

lm=sm.OLS(y_train,x_train_lm).fit()

print(lm.summary())


# # References

# Dataset - https://archive.ics.uci.edu/dataset/9/auto+mpg
# https://realpython.com/linear-regression-in-python/ 
# https://medium.com/@hackdeploy/python-linear-regression-analysis-7b3cfb01a748
# https://realpython.com/linear-regression-in-python/
# https://towardsdatascience.com/how-to-simplify-hypothesis-testing-for-linear-regression-in-python-8b43f6917c86
# https://medium.com/nerd-for-tech/hypothesis-testing-on-linear-regression-c2a1799ba964
