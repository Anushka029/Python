#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/Anushka029/Datasets/main/concrete.csv")


# In[3]:


df.sample(10)


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.describe(include='all').T


# In[9]:


sns.boxplot(x=df['strength'])


# In[10]:


sns.pairplot(df,diag_kind='kde')


# In[11]:


corr=df.corr()
corr


# In[12]:


sns.heatmap(corr,annot=True, cmap='twilight')


# ## Removing outliers

# In[13]:


def remove_outlier(col):
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    iqr=q3-q1
    lower_range=q1-(1.5*iqr)
    upper_range=q3+(1.5*iqr)
    return lower_range, upper_range


# In[14]:


lowleadtime, uppleadtime=remove_outlier(df['strength'])
df['strength']=np.where(df['strength']>uppleadtime, uppleadtime, df['strength'])
df['strength']=np.where(df['strength']<lowleadtime, lowleadtime, df['strength'])


# In[16]:


sns.boxplot(df['strength'])


# In[17]:


sns.boxplot(df['cement'])


# In[19]:


Y=df['strength']
X=df.drop(['strength'], axis=1)


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[21]:


model1=LinearRegression()


# In[22]:


model1.fit(X_train, Y_train)


# In[23]:


model1.score(X_train, Y_train)


# In[24]:


model1.score(X_test, Y_test)


# In[ ]:




