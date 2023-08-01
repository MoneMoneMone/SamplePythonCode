#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Data Analysis of a Salesforce Data


# In[ ]:


#This is a basic python which will show analysis on salesforce data. I retreived the data from google datasearch and kaggle.com.
#Since this is basic python, only few libraries are used for data analysis purposes. 


# In[ ]:


#Numpy is needed to manipulate data (i.e. change values in cell, update values, or replace values). 
#Numpy is not only limited #to cell but also lists and tupples. Calculations are also done in numpy

#Pandas is used to describe the data set and show statistical analysis of the data

#Seaborn is used to provide a visual representation of the data in order to gather insight easily.

#statsmodels and linear regression libraries are import for linear regression of data.


# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sn
from sklearn.linear_model import LinearRegression

salesforce1 = pd.read_csv("C:/Users/ADMIN/Desktop/Edited/Salesforce data 1.csv", header = 0)


# In[ ]:


#The first thing to do is describe the data to have an insight on on every parameter. You can see average customer age and corresponding average revenue. It also shows percentile for different columns.#


# In[12]:


salesforce1.describe()


# In[9]:


salesforce1.head


# In[ ]:


#In order to check certain indices in the table, the syntax below is used


# In[10]:


salesforce1[0:5]


# In[ ]:


#iloc is used to determine indices inside the bracket, []. The difference from the above syntax is that iloc includes the last value


# In[11]:


salesforce1.iloc[10:15]


# In[ ]:


#It is not practical to see values with more than 2 decimal place so I reduce it to 2.
pd.set_option("display.precision", 2)


# In[25]:


salesforce1.loc[0:5,'Revenue']


# In[30]:


#Keep in mind that python is a case sensitive program. Thus, attention to detail is a must!
#Example below is shown.


# In[31]:


salesforce1['revenue'].mean()


# In[33]:


salesforce1['Revenue'].mean()


# In[44]:


#You can determine the frequency of a certain value by inputting the syntax below
salesforce1['Customer Gender'].value_counts()


# In[45]:


salesforce1['Customer Age'].value_counts()


# In[47]:


#Pivot table in Jupyter Notbook
salesforce1.pivot_table(values='Revenue', index = 'Customer Age')


# In[48]:


salesforce1.pivot_table(values='Revenue', index= 'Customer Age', aggfunc=[np.mean, np.median])


# In[49]:


salesforce1.pivot_table(values='Revenue', index = 'Customer Age', columns = 'Customer Gender')
# AS shown below, there are missing values, these values are replaced with mean


# In[78]:


salesforce1.pivot_table(values='Revenue', index = 'Customer Age', columns = 'Customer Gender', fill_value = salesforce1.Revenue.mean())


# In[63]:


#To Show the data based on a certain column as an index, the syntax below is used
sf_ind = salesforce1.set_index('Country','State') 
sf_ind


# In[62]:


sf_ind['Revenue'].max()


# In[68]:


salesforce1.reset_index()


# In[69]:


sf_sort = salesforce1.sort_index()


# In[72]:


sf_ind = salesforce1.set_index('State') 
print(sf_sort.loc['Washington':'Hauts de Seine'])


# # Seaborn

# In[80]:


#To plot the Revenue column in the Salesforce1 data, I used the seaborn library
sns.distplot(salesforce1.Revenue, kde = False)


# In[81]:


np.percentile(salesforce1.Revenue,[99])


# In[85]:


sf1 = salesforce1.copy()


# In[86]:


sf1 = sf1.fillna(sf1.mean())


# In[88]:


#I noticed that you can calculate the profit gain from each index by subtracting Cost from Revenue
sf1['Profit'] = sf1.Revenue - sf1.Cost
sf1.head()


# In[89]:


sf1.describe()


# In[90]:


#Dummies is used to give value to descriptive type of data. In this case (0 and 1) are used for binary purposes
#This is for the preparation of matplot
sf1 = pd.get_dummies(sf1)
sf1


# In[103]:


#In matplot, adding constant 1 is important for the linear regression analysis
X = sn.add_constant(sf1['Customer Age'])
lm = sn.OLS(sf1['Profit'],X).fit()
X.head()


# In[94]:


lm.summary()


# In[133]:


x = sf1['Sub Category_Tires and Tubes']
y = sf1['Profit']


# In[135]:


#Based on the linear regression, X intercept is 58.41 while y intercept is 0.177
lm2 = LinearRegression()
lm2.fit(X,y)
print(lm2.intercept_, lm2.coef_)


# In[107]:


#For more syntax regarding linear regression
help(lm2)


# In[116]:


sns.jointplot(x=sf1['Customer Age'], y=sf1['Profit'], data =sf1, kind ='reg')


# In[121]:


sns.pairplot(sf1, x_vars=['Customer Age'], y_vars='Profit', size=5, aspect=1.0, kind='reg')


# In[ ]:


#So far, these are the codes that I learned when I was studying basic Python

