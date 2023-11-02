#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


# In[43]:


train_data = pd.read_csv('D:\\Downloads\\archive\\fraudTrain.csv')
test_data = pd.read_csv('D:\\Downloads\\archive\\fraudTest.csv')
print("Info of Train Data:")
print(train_data.info())
print("\nInfo of Test Data Info:")
print(test_data.info())


# In[44]:


train_data.head()


# In[45]:


train_data.describe()


# In[46]:


test_data.describe()


# In[3]:


# checking for missing data in both the dataset
print("Train data-> missing value")
print(train_data.isnull().sum())
print("-----------------------------------------------")
print("\nTest data-> missing value")
print(test_data.isnull().sum())


# In[4]:


# Visualize the distribution of the target variable (fraudulent or not)
plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Distribution of Fraudulent Transactions')
plt.xlabel(' (0:Not Fraud  | 1:Fraud) ')
plt.ylabel('Count')
plt.show()


# In[5]:


print(test_data['is_fraud'].value_counts())
print(train_data['is_fraud'].value_counts())


# In[6]:


# Removing rows with missing values
# because it's just single row in each set,
# that's why there will no huge data loss.
cols_to_drop = ['Unnamed: 0','cc_num','merchant','first','last','trans_num','unix_time','street','category']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)


# In[7]:


cols_to_drop = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','state']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)


# In[11]:


train_data.is_fraud.value_counts()


# In[12]:


train_data.gender =[ 1 if value == "M" else 0 for value in train_data.gender]
test_data.gender =[ 1 if value == "M" else 0 for value in test_data.gender]


# In[13]:


train_data.head()


# In[42]:


legit = train_data[train_data.is_fraud == 0]
fraud = train_data[train_data.is_fraud == 1]


# In[21]:


fraud.shape


# In[22]:


legit.shape


# In[ ]:


legit_sample = legit.sample(n=len(fraud), random_state=2)
train_data = pd.concat([legit_sample, fraud], axis=0)


# In[ ]:


train_data['is_fraud'].value_counts()


# In[29]:


# split data into training and testing set

X = train_data.drop('is_fraud',axis=1)
# X_test = test_data.drop('is_fraud',axis=1)
Y = train_data['is_fraud']
# y_test = test_data['is_fraud']


# In[30]:


X.shape


# In[31]:


Y.shape


# In[32]:


# storing 80% data in X and Y train for training the data and the remaining 20% data is for testing
# stratify we are using when the value of X_train and Y_train as well as X_test and Y_test are same respectively 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


# In[33]:


#  logistic regression-> classify the true and false i.e 0 and 1
# creating object of LR model-> train LR model
model = LogisticRegression()
model.fit(X_train, Y_train)
ypred = model.predict(X_test) 
# comparision of 20% of Y_test and 20% of ypred that model are prepared on the basis of 80% X_train and Y_train 


# In[41]:


accuracy_score(ypred, Y_test)

