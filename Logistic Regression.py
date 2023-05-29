#!/usr/bin/env python
# coding: utf-8

# # To find whether passenger survived or not

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/Anushka029/Datasets/main/titanic-training-data.csv")


# In[3]:


df.sample(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe(include='all').T


# In[8]:


df.dtypes


# In[10]:


median1=df['Age'].median()
median1


# In[13]:


df['Age'].replace(np.nan, median1, inplace=True)


# In[14]:


df.isnull().sum()


# In[19]:


df.drop('Name',axis=1, inplace=True)


# In[20]:


df.drop('Ticket',axis=1, inplace=True)


# In[21]:


df.drop('Fare',axis=1, inplace=True)


# In[22]:


df.drop('Cabin',axis=1, inplace=True)


# In[23]:


df.shape


# In[24]:


df.sample(10)


# In[25]:


df.isnull().sum()


# In[32]:


mode1=df['Embarked'].mode().values[0]


# In[33]:


mode1


# In[34]:


df['Embarked'].replace(np.nan, mode1, inplace=True)


# In[35]:


df.isnull().sum()


# In[36]:


df=pd.get_dummies(df,columns=['Embarked'])


# In[37]:


df.sample(10)


# In[38]:


df=pd.get_dummies(df,columns=['Sex'])


# In[39]:


df.sample(10)


# In[40]:


Y=df['Survived']
X=df.drop(['Survived'], axis=1)


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[44]:


model1=LogisticRegression()


# In[45]:


model1.fit(X_train, Y_train)


# In[46]:


model1.score(X_train, Y_train)


# In[47]:


model1.score(X_test, Y_test)


# In[ ]:




