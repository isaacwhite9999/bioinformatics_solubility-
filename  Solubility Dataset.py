#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


delaney_with_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(delaney_with_descriptors_url)
dataset


# In[3]:


X = dataset.drop(['logS'], axis=1)
X


# In[4]:


Y = dataset.iloc[:,-1]
Y


# In[5]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[6]:


model = linear_model.LinearRegression()
model.fit(X, Y)


# In[7]:


Y_pred = model.predict(X)
Y_pred


# In[8]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y, Y_pred))


# In[9]:


print('LogS = %.2f %.2f LogP %.4f MW + %.4f RB %.2f AP' % (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3] ) )


# In[10]:


import matplotlib.pyplot as plt
import numpy as np


# In[11]:


import pickle


# In[12]:


pickle.dump(model, open('solubility_model.pkl', 'wb'))


# In[ ]:




