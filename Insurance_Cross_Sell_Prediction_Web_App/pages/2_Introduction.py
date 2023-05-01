import numpy as np
import pandas as pd
import streamlit as st

st.title("Introduction")

st.markdown("""
The goal is to predict whether health insurance owners' from past year would also be interested in purchasing vehicle 
insurance coverage provided by the Company.

In the first step, the cross-selling prediction activity, have been used calibrated classifiers. Logistic Regression 
has been employed as a benchmark model and it has been compared with other machine learning models such as Naïve Bayes and Hist 
Gradient Boosting Machine. The ensemble model shows the best performance. 

In the second step, has been profiled customers interested in the purchasing coverage using K-means clustering method on numerical features 
and then applying the split to the all data set. In this way has been possible to understand relationships between 
numerical features as Annaul Premium and Age with categorical features. 

### Data Description

#### Variables

**id**

Unique ID for the customer

**Gender**

Gender of the customer

**Age**

Age of the customer

**Driving_License**

0 : Customer does not have DL 

1 : Customer already has DL

**Region_Code**

Unique code for the region of the customer

**Previously_Insured**

1 : Customer already has Vehicle Insurance 

0 : Customer doesn't have Vehicle Insurance

**Vehicle_Age**

Age of the Vehicle

**Vehicle_Damage**

1 : Customer got his/her vehicle damaged in the past 

0 : Customer didn't get his/her vehicle damaged in the past

**Annual_Premium**

The amount customer needs to pay as premium in the year

**PolicySalesChannel**

Anonymized Code for the channel of outreaching to the customer ie. 
Different Agents, Over Mail, Over Phone, In Person, etc.

**Vintage**

Number of Days, Customer has been associated with the Company

**Response**

1 : Customer is interested 

0 : Customer is not interested


#### Evaluation Metric

The evaluation metric used for this data set is ROC_AUC score

* Data set source: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

""")
