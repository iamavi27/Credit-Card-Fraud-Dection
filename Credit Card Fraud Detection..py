import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



train_data = pd.read_csv('D:\\Downloads\\archive\\fraudTrain.csv')
test_data = pd.read_csv('D:\\Downloads\\archive\\fraudTest.csv')
print("Info of Train Data:")
print(train_data.info())
print("\nInfo of Test Data Info:")
print(test_data.info())

train_data.head()

train_data.describe()

test_data.describe()

# checking for missing data in both the dataset
print("Train data-> missing value")
print(train_data.isnull().sum())
print("-----------------------------------------------")
print("\nTest data-> missing value")
print(test_data.isnull().sum())

# Visualize the distribution of the target variable (fraudulent or not)
plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Distribution of Fraudulent Transactions')
plt.xlabel(' (0:Not Fraud  | 1:Fraud) ')
plt.ylabel('Count')
plt.show()


print(test_data['is_fraud'].value_counts())
print(train_data['is_fraud'].value_counts())


# Removing rows with missing values
# because it's just single row in each set,
# that's why there will no huge data loss.
cols_to_drop = ['Unnamed: 0','cc_num','merchant','first','last','trans_num','unix_time','street','category']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)


cols_to_drop = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','state']
train_data.drop(columns=cols_to_drop,inplace = True)
test_data.drop(columns=cols_to_drop,inplace = True)


train_data.is_fraud.value_counts()


train_data.gender =[ 1 if value == "M" else 0 for value in train_data.gender]
test_data.gender =[ 1 if value == "M" else 0 for value in test_data.gender]


train_data.head()


legit = train_data[train_data.is_fraud == 0]
fraud = train_data[train_data.is_fraud == 1]

fraud.shape


legit.shape

legit_sample = legit.sample(n=len(fraud), random_state=2)
train_data = pd.concat([legit_sample, fraud], axis=0)

train_data['is_fraud'].value_counts()


# split data into training and testing set

X = train_data.drop('is_fraud',axis=1)
# X_test = test_data.drop('is_fraud',axis=1)
Y = train_data['is_fraud']
# y_test = test_data['is_fraud']

X.shape


Y.shape


# storing 80% data in X and Y train for training the data and the remaining 20% data is for testing
# stratify we are using when the value of X_train and Y_train as well as X_test and Y_test are same respectively 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


#  logistic regression-> classify the true and false i.e 0 and 1
# creating object of LR model-> train LR model
model = LogisticRegression()
model.fit(X_train, Y_train)
ypred = model.predict(X_test) 

# comparision of 20% of Y_test and 20% of ypred that model are prepared on the basis of 80% X_train and Y_train 

accuracy_score(ypred, Y_test)

