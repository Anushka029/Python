#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/Anushka029/Datasets/main/hotel_bookings.csv")


# In[3]:


df.sample(10)


# In[4]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[11]:


median1=df['company'].median()
median1


# In[12]:


df['company'].replace(np.nan,median1,inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


mean1=df['children'].mean()


# In[15]:


mean1


# In[16]:


df['children'].replace(np.nan,mean1,inplace=True)


# In[17]:


df.isnull().sum()


# In[20]:


mode1=df['country'].mode().values[0]
mode1


# In[21]:


df['country'].replace(np.nan,mode1,inplace=True)


# In[22]:


df.isnull().sum()


# ## Removing attributers

# In[23]:


df.drop('agent',axis=1, inplace=True)


# In[24]:


df.shape


# ## Removing duplicates

# In[25]:


duplicate=df.duplicated()


# In[26]:


duplicate.sum()


# In[27]:


df.boxplot('lead_time')


# ## Remove outliers

# In[31]:


def remove_outlier(col):
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    iqr=q3-q1
    lower_range=q1-(1.5*iqr)
    upper_range=q3+(1.5*iqr)
    return lower_range, upper_range


# In[32]:


lowleadtime, uppleadtime=remove_outlier(df['lead_time'])
df['lead_time']=np.where(df['lead_time']>uppleadtime, uppleadtime, df['lead_time'])
df['lead_time']=np.where(df['lead_time']<lowleadtime, lowleadtime, df['lead_time'])


# In[33]:


df.boxplot('lead_time')


# In[34]:


df.boxplot('is_canceled')


# In[35]:


df.boxplot('babies')


# In[36]:


def remove_outlier(col):
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    iqr=q3-q1
    lower_range=q1-(1.5*iqr)
    upper_range=q3+(1.5*iqr)
    return lower_range, upper_range


# In[37]:


lowleadtime, uppleadtime=remove_outlier(df['babies'])
df['babies']=np.where(df['babies']>uppleadtime, uppleadtime, df['babies'])
df['babies']=np.where(df['babies']<lowleadtime, lowleadtime, df['babies'])


# In[38]:


df.boxplot('babies')


# In[ ]:




