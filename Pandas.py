#!/usr/bin/env python
# coding: utf-8

# ### Pandas

# In[5]:


import numpy as np
import pandas as pd


# In[11]:


df=pd.read_csv("titanic-training-data.csv")


# In[12]:


df.head()


# In[13]:


df.head(10)


# In[15]:


df.sample()


# In[16]:


df.dtypes


# In[17]:


df.info()


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[20]:


df.describe()


# In[21]:


df.describe(include="all")


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


plt.hist(x=df["Age"])


# In[37]:


plt.hist(x=df["Age"],color='Black')
plt.title("Distribution of Age", color='blue', fontsize=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[44]:


sns.boxplot(x=df["Age"])
plt.title("Distribution of Age", color='blue', fontsize=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[50]:


df["Sex"].value_counts().plot(kind="bar")


# In[49]:


plt.hist(x=df["Pclass"],color='Black')
plt.title("Distribution of Pclass", color='blue', fontsize=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[64]:


from matplotlib import cm
emb=df["Embarked"].value_counts()
keys=emb.keys().to_list()
counts=emb.to_list()
cs=cm.Set1([4,6,8,10])
plt.pie(x=counts, labels=keys, autopct='%1.1f%%',colors=cs)
plt.show()


# In[56]:


plt.scatter(x="Age", y="SibSp", data=df)


# In[ ]:




