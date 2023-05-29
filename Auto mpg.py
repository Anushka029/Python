#!/usr/bin/env python
# coding: utf-8

# # Understanding relationship between mpg and other attributes

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


df=pd.read_csv("https://raw.githubusercontent.com/Anushka029/Datasets/main/auto-mpg.csv")


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.head()


# In[10]:


sns.boxplot(x=df['mpg'])


# In[11]:


df.hist(figsize=(10,20))


# In[12]:


sns.boxplot(x="mpg", data=df)


# In[13]:


sns.pairplot(df,diag_kind='kde')


# In[14]:


corr=df.corr()
corr


# In[15]:


sns.heatmap(corr,annot=True, cmap='twilight')


# In[16]:


sns.boxplot(x='mpg',data=df)


# In[17]:


duplicate=df.duplicated()


# In[18]:


duplicate.sum()


# In[19]:


df.drop('car name',axis=1, inplace=True)


# In[20]:


df.head()


# ## Dealing with missing values

# In[21]:


df['horsepower']=df['horsepower'].replace('?', np.nan)


# In[22]:


df['horsepower']=df['horsepower'].astype(float)


# In[23]:


median1=df['horsepower'].median()
median1


# In[24]:


df['horsepower'].replace(np.nan,median1,inplace=True)


# In[25]:


df.dtypes


# ## Replacing origin with country names

# In[26]:


df['origin']=df['origin'].replace({1:'America', 2:'Europe', 3:'Asia'})


# In[27]:


df.head()


# In[28]:


df.sample(10)


# ## Encoding

# In[29]:


df=pd.get_dummies(df,columns=['origin'])


# In[30]:


df.sample(10)


# In[31]:


df.dtypes


# In[32]:


sns.pairplot(df)


# In[33]:


Y=df['mpg']
X=df.drop(['mpg'], axis=1)


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[38]:


model1=LinearRegression()


# In[39]:


model1.fit(X_train,Y_train)


# In[40]:


model1.score(X_train, Y_train)


# In[41]:


model1.score(X_test, Y_test)


# In[ ]:




