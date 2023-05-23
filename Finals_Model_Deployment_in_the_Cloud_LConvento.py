#!/usr/bin/env python
# coding: utf-8

# # Leiann Convento
# # Final Exam: Model Deployment in the Cloud
# 
# **Objective(s):**
# 
# This activity aims to apply all the learnings for the Final Period. 
# 
# **Intended Learning Outcomes (ILOs):**
# 
# Demonstrate how to train and save a model.
# Demonstrate how to deploy the deep learning model in the cloud. (not Machine Learning model) 
#  
# 
# **Instructions:**
# 
# You can choose any previous deep learning model. 
# Follow the instructions on deploying a model using Streamlit App in the cloud. 
#  
# 
#  
# 
# **Note:** An accessible URL of the APP should be submitted. Also, upload the Github repo link. Strictly no straight copying from the internet. 

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path="/Users/leiannconvento/Documents/ADS/Finals/disaster_tweets_model.h5"
model=load_model(model_path)

# Function to process the user input
def process_text(input_text):
    tokenizer=Tokenizer(num_words=10000)
    X_train=pd.read_csv('X_train.csv')
    tokenizer.fit_on_texts(X_train['text'])
    max_sequence_length=100
    tweet=[input_text]
    newseq1=tokenizer.texts_to_sequences(tweet)
    newdata1=pad_sequences(newseq1,maxlen=max_sequence_length)
    newpredict1=np.round(model.predict(newdata1)).flatten()
    label = "This tweet is about Disasters" if newpredict1[0] == 1 else "This tweet is not about Disasters"
    return label

#Streamlit
st.title("PSMDS Advance Data Science Final Project: Disaster Tweets Identifier")
st.text("This model will identify if the input tweet is about disasters")
st.write("Type in your tweet:")

#Streamlit input box
user_input=st.text_area("","")

#Streamlit Identifying Tweet
if st.button("Identify Tweet"):
    if user_input:
        processed_result=process_text(user_input)
        st.write("This tweet is:")
        st.write(processed_result)
    else:
        st.write("Please type in your tweet")

