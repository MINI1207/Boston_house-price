#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# ### IMPORTING some toy dataset from sklearn

# In[3]:


from sklearn.datasets import load_boston


# In[4]:


loaded_data = load_boston()
print( loaded_data.DESCR )


# In[5]:


loaded_data.data.shape


# In[6]:


loaded_data.target.shape


# In[7]:


data = pd.DataFrame(loaded_data.data,columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"])
data["MEDV"] = loaded_data.target
data.head()


# In[8]:


data.to_csv("boston.csv")


# In[9]:


data[ (data.AGE<20) ][ data.RAD==3.0]


# In[10]:


data.describe()


# ### findoutthe avg cost of house where crime rate is below .08
# ### findout avg cost of the house where crime rate is above 3.6

# In[11]:


data_with_crim_lower=data[ data.CRIM <0.08]
data_with_crim_lower.MEDV.mean()


# In[12]:


data_with_crim_higher=data[ data.CRIM > 3.6]
data_with_crim_higher.MEDV.mean()


# ### find out if most expensive house lie underlower crime rate or high crime rate

# In[13]:


data.MEDV.max()


# In[14]:


data_with_crim_lower[data.MEDV>=30].shape[0]


# In[15]:


data_with_crim_higher[data.MEDV>=30].shape[0]


# In[ ]:




