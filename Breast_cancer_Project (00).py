#!/usr/bin/env python
# coding: utf-8

# # **Loading the data into python**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('data.csv')


# In[ ]:


df.head()


# ## **PERFORMING EDA**

# In[ ]:


df.isna().sum()


# In[ ]:


df = df.dropna(axis=1)


# In[ ]:


df.info()


# In[ ]:


# Dependent variable or Output variable
df['diagnosis'].value_counts()


# In[ ]:


#Now, B and M is Object datatype 
sns.countplot(df['diagnosis'],label='count')


# In[ ]:


df.dtypes


# In[ ]:


#Now, Encode x and y into 0 and 1 (which is In Binary)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.iloc[:,1]=le.fit_transform(df.iloc[:,1].values)


# In[ ]:


df.head()


# In[ ]:


# Now, Check unique value in diagnosis columm
df['diagnosis'].unique()


# In[ ]:


data=df.iloc[:,1:11]  
data


# In[ ]:


# Creating histogram without grid
histogram =data.hist(bins=10,figsize=(15,10),grid=False)


# In[ ]:


#Now, Create histogram with grid
histogram =data.hist(bins=10,figsize=(15,10),grid=True)


# In[ ]:


#Now Create heatmap
plt.figure(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,fmt='.0%')  


# In[ ]:


#Here, Density plot
plt=data.plot(kind='density',subplots=True,layout=(4,3),sharex=False,sharey=False,
              fontsize=12,figsize=(15,10))  


# # **Training the data by a classification models**

# In[ ]:


# Now split the dataset
from sklearn.model_selection import train_test_split


# In[ ]:


x=df.drop(['diagnosis'],axis=1)
y=df['diagnosis'].values


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=0)


# In[47]:


# Now Check accuracy with logistic regression
from sklearn.linear_model import LogisticRegression
reg= LogisticRegression()
reg.fit(x_train,y_train)
print("Logistic Regression accuracy :{:.2f}%".format(reg.score(x_test,y_test)*100))


# In[ ]:


#Now SVM Classifier
from sklearn.svm import SVC
svm= SVC(random_state=1)
svm.fit(x_train,y_train)
print("SVC accuracy : {:.2f}%".format(svm.score(x_test,y_test)*100))


# In[ ]:


#Now naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
print("Naive Bayes accuracy: {:.2f}%".format(nb.score(x_test,y_test)*100))


# In[ ]:


#Now Random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
print("Random Forest Classifier accuracy: {:.2f}%".format(rf.score(x_test,y_test)*100))


# In[ ]:


#xg boost
import xgboost
xg=xgboost.XGBClassifier()
xg.fit(x_train,y_train)
print("XGboost accuracy: {:.2f}%".format(xg.score(x_test,y_test)*100))


# **RANDOM FOREST & xgboost have accuracy>=80%**

# # **Making prediction using single test record**

# In[ ]:


# Now use the trained model to make prediction
# Exactly we need to pass similar or same format of data to Predict on which M.L model learnt
x_test


# In[ ]:


x_test.head()


# In[ ]:


rf.predict(x_test)


# In[ ]:


np.array(y_test)


# In[ ]:


#Now prediction on test data to predict if someone has Breast cancer!


# In[ ]:


# Now comparing our predictions to truth labels to evaluate the model
y_preds= rf.predict(x_test)
np.mean(y_preds==y_test)


# In[ ]:


rf.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_preds)

