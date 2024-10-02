#!/usr/bin/env python
# coding: utf-8

# #  Data science automation

# This week is all about looking at automation tehcniques for data science and with Python. We can automate a lot of things with Python: collecting data, processing it, cleaning it, and many other parts of the data science pipeline. Here, we will show how to:
# 
# - use the pycaret autoML Python package to find an optimized ML model for our diabetes dataset
# - create a Python script to ingest new data and make predictions on it
# 
# Often, next steps in fully operationalizing an ML pipeline like this are to use a cloud service to scale and serve our ML algorithm. We can use things like AWS lambda, GCP, AWS, or Azure ML depolyment with tools such as docker and kubernetes.

# # Data Preparation

# It includes pulling the dataset from the system and we are going to load our same prepared data from week 2 where everything has been converted to numbers.

# In[38]:


import pandas as pd


df = pd.read_csv(r"C:\Users\sai teja\Downloads\prepared_churn_data.csv" ,index_col='customerID')
df


# In[39]:


from pycaret.classification import *


# In[40]:


automl = setup(df, target='Churn')


# INTERPRETATION:
# 
# Here, the preprocess is true it includes outliers treatment ,missing value treatment and feature engineering

# In[41]:


best_model = compare_models()


# INTERPRETATION:
# 
# 1. Here, every model is being tested to get best accuracy by automl.
# 2. Gradient Boosting Classifier is the best model for given dataset.

# In[42]:


best_model


# INTERPRETATION:
# 
# 1. These are the best parameters for the model after hyperparameter tuning.

# In[43]:


df.iloc[-2:-1]


# We are selecting the last row, but using the indexing `[-2:-1]` to make it a 2D array instead of 1D (which throws an error). Try running `df.iloc[-1].shape` and `df.iloc[-2:-1].shape` to see how they differ.
# 
# However, this only works if we set `preprocess=False` in our setup function. Otherwise the order of features may be different
# 
# A more robust way (in case we are using preprocessing with autoML) is to use pycaret's predict_model function:

# In[44]:


predict_model(best_model, df.iloc[-2:-1])


# In[45]:


plot_model(best_model, plot='auc')


# INTERPRETATIONS:
# 1. In this plot, The roc of class 0 and roc of class 1 both are positivily increasing against false positive rate and towards true positive rate which proves the efficiency of prediction.
# 2. the AUC values of class 0 and 1 are above 0.8 which indicates better discrimination performance of the model.

# In[46]:


plot_model(best_model, plot='pr')


# INTERPRETATION:
# 1. The Precision-Recall (PR) curve is another evaluation metric used in binary classification tasks, particularly when dealing with imbalanced datasets.
# 2. The PR curve is a graphical representation of the trade-off between precision and recall for different threshold values used to classify instances as positive or negative.
# 3. the curve has  downward trend i.e; with increase in Recall the Precision is decreasing rapidly.

# # SAVING AND LOADING MODEL

# Next, we want to save our trained model so we can use it in a Python file

# In[47]:


save_model(best_model, 'lr')


# In[48]:


import pickle
with open('lr.pkl','wb') as f:
    pickle.dump(best_model, f)


# In[49]:


with open('lr.pkl','rb') as f:
    loaded_model = pickle.load(f)


# In[50]:


new_data=df.iloc[-2:-1]


# In[51]:


predict_model(loaded_model, new_data)


# INTERPRETATION:
# 1. The prediction score is around 58%

# # Making a Python module to make predictions

# In[60]:


df = pd.read_csv(r"C:\Users\sai teja\Downloads\prediction.py", delimiter=';')


# In[61]:


df


# In[62]:


from IPython.display import Code

Code(r"C:\Users\sai teja\Downloads\prediction.py")


# In[63]:


get_ipython().run_line_magic('run', '"C:\\Users\\sai teja\\Downloads\\prediction.py"')


# # SUMMARY

# First, a telecoms churn dataset with solely numerical variables was utilized to train and test the machine learning model.
# Next, I imported the Pycaret package, which contains the 'automl' function. This function performs preprocessing by default, splitting data into training and testing sets, and then loads the training and testing sets into each model. It also tunes hyperparameters, providing the best machine learning model with the highest accuracy and the minimum number of parameters needed.
# We may load, predict data, and save the model using the corresponding functions defined in the 'pycaret' package by using the information obtained by the 'automl' function.
# I made a Python module and linked it to the current directory or notebook using the 'code()' and 'run' functions from the 'IPython.display' package in order to verify the predictions' accuracy.
# For my forecast, I used a fresh churn dataset with known values.
# I have saved and uploaded every file utilized and created into the GITHUB after projecting and evaluating efficiency.

# In[ ]:




