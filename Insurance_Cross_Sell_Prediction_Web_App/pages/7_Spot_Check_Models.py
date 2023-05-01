import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import pickle5 as pickle


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Spot Check Models")

st.markdown("""
Logistic Regression (LR) is used as a benchmark model, because it is usually used as reference in Insurance, and also because 
in terms of calibration for the evaluation it is a calibrated model. 

For this job Logistic Regression (LR) has been compared with Gaussian Naive Bayes model (GNB), and Histogram-based Gradient Boosting Machine (HGBM).

Each model is evaluated if it is properly calibrated, and eventually, is applied the Platt Scaling to perform a well-calibrated classifier.

""")


DATA_URL_xtr = ('./data/X_train.csv')
X_train = pd.read_csv(DATA_URL_xtr)
DATA_URL_xte = ('./data/X_test.csv')
X_test = pd.read_csv(DATA_URL_xte)
DATA_URL_ytr = ('./data/y_train.csv')
y_train = pd.read_csv(DATA_URL_ytr)
DATA_URL_yte = ('./data/y_test.csv')
y_test = pd.read_csv(DATA_URL_yte)

# LR model
# loading in the model to predict on the data
with open('./data/LR_classifier.pkl', 'rb') as pickle_in:
    LR_classifier = pickle.load(pickle_in)
    
predictions_tr = LR_classifier.predict_proba(X_train)[:, 1]
predictions_t = LR_classifier.predict_proba(X_test)[:, 1]
LR_auc_train = roc_auc_score(y_train, predictions_tr)  
LR_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['LR'], 'auc_train':[LR_auc_train],'auc_test':[LR_auc_test]}
LR_score= pd.DataFrame(score)

# GNB model
# loading in the model to predict on the data
with open('./data/GNB_classifier_.pkl', 'rb') as pickle_in:
    GNB_classifier = pickle.load(pickle_in)
    
predictions_tr = GNB_classifier.predict_proba(X_train)[:, 1]
predictions_t = GNB_classifier.predict_proba(X_test)[:, 1]
GNB_auc_train = roc_auc_score(y_train, predictions_tr)  
GNB_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['GNB'], 'auc_train':[GNB_auc_train],'auc_test':[GNB_auc_test]}
GNB_score= pd.DataFrame(score)

# HGBM model
# loading in the model to predict on the data
with open('./data/HGBM_classifier.pkl', 'rb') as pickle_in:
    HGBM_classifier = pickle.load(pickle_in)
    
predictions_tr = HGBM_classifier.predict_proba(X_train)[:, 1]
predictions_t = HGBM_classifier.predict_proba(X_test)[:, 1]
HGBM_auc_train = roc_auc_score(y_train, predictions_tr)  
HGBM_auc_test = roc_auc_score(y_test, predictions_t) 
score= {'model':['HGBM'], 'auc_train':[HGBM_auc_train],'auc_test':[HGBM_auc_test]}
HGBM_score= pd.DataFrame(score)

score_cal = LR_score.append(GNB_score)
score_cal = score_cal.append(HGBM_score)
score_cal

# Plot results for a graphical comparison
plt.rcParams['figure.figsize']=(15,5)
fig = plt.figure()
plt.subplot(1,2,1)  
sns.stripplot(x="model", y="auc_train",data=score_cal,size=15)
plt.xticks(rotation=45)
plt.title('Train results')
axes = plt.gca()
axes.set_ylim([0,1.1])
plt.subplot(1,2,2)
sns.stripplot(x="model", y="auc_test",data=score_cal,size=15)
plt.xticks(rotation=45)
plt.title('Test results')
axes = plt.gca()
axes.set_ylim([0,1.1])
st.pyplot(fig)

st.subheader("Check Calibration Models")

# check LR calibration
# Generate probability predictions from your model
probabilities = LR_classifier.predict_proba(X_test)
predicted_probabilities = probabilities[:, 1]

# Get true outcome value for each test observation
test_outcomes = y_test

# Generate the calibration curve data
calibration_curve_data = calibration_curve(test_outcomes, predicted_probabilities, n_bins=10)

# Plot the calibration curve
plt.rcParams['figure.figsize']=(5,2.5)
fig = plt.figure()
plt.plot(calibration_curve_data[1], calibration_curve_data[0], marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('LR Calibration Curve')
st.pyplot(fig)

# check GNB calibration
# Generate probability predictions from your model
probabilities = GNB_classifier.predict_proba(X_test)
predicted_probabilities = probabilities[:, 1]

# Get true outcome value for each test observation
test_outcomes = y_test

# Generate the calibration curve data
calibration_curve_data = calibration_curve(test_outcomes, predicted_probabilities, n_bins=10)

# Plot the calibration curve
plt.rcParams['figure.figsize']=(5,2.5)
fig = plt.figure()
plt.plot(calibration_curve_data[1], calibration_curve_data[0], marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('GNB Calibration Curve')
st.pyplot(fig)

# check HGBM calibration
# Generate probability predictions from your model
probabilities = HGBM_classifier.predict_proba(X_test)
predicted_probabilities = probabilities[:, 1]

# Get true outcome value for each test observation
test_outcomes = y_test

# Generate the calibration curve data
calibration_curve_data = calibration_curve(test_outcomes, predicted_probabilities, n_bins=10)

# Plot the calibration curve
plt.rcParams['figure.figsize']=(5,2.5)
fig = plt.figure()
plt.plot(calibration_curve_data[1], calibration_curve_data[0], marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('HGBM Calibration Curve')
st.pyplot(fig)


