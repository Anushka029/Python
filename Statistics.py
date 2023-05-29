#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


mydata=pd.read_csv("CardioGoodFitness-1.csv")


# In[3]:


mydata.sample(10)


# In[4]:


mydata.shape


# In[5]:


mydata.dtypes


# In[6]:


mydata.info()


# In[7]:


mydata.describe()


# In[8]:


mydata.describe(include="all").T


# In[11]:


sns.boxplot(x=mydata["Age"])


# In[12]:


sns.countplot(x="Product", hue="Gender", data=mydata)


# In[13]:


sns.countplot(x="Product", hue="MaritalStatus", data=mydata)


# In[14]:


mydata.hist(figsize=(10,20))


# In[15]:


sns.boxplot(x="Product", y= "Age", data=mydata)


# In[16]:


pd.crosstab(mydata['Product'], mydata['Gender'])


# In[17]:


sns.pairplot(mydata,diag_kind='kde')


# In[19]:


corr=mydata.corr()
corr


# In[30]:


sns.heatmap(corr,annot=True, cmap='twilight')


# In[ ]:




