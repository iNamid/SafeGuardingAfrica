#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# In[50]:


data = pd.read_csv('child_malnutrition_africa.csv')

categorical_cols = ['ISO code', 'Continent', 'World Bank']
numeric_cols = ['Sex', 'Age', 'Height', 'Weight', 'Wasting', 'Overweight', 'Stunting', 'Underweight']


# In[51]:


numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])


# In[52]:


categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])


# In[53]:


label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])


# In[54]:


X = data.drop('Status', axis=1)
y = data['Status']


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


model = RandomForestClassifier()


# In[57]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[60]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


# In[59]:


print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

