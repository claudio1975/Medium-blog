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
import shap
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


st.title("Features Importance")

DATA_URL_xtr = ('./data/X_train.csv')
X_train = pd.read_csv(DATA_URL_xtr)
DATA_URL_xte = ('./data/X_test.csv')
X_test = pd.read_csv(DATA_URL_xte)
DATA_URL_ytr = ('./data/y_train.csv')
y_train = pd.read_csv(DATA_URL_ytr)
DATA_URL_yte = ('./data/y_test.csv')
y_test = pd.read_csv(DATA_URL_yte)

st.markdown("""
The most relevant feature with impact on the target variable is "Previously_Insured".
""")

# load shap_values:
shap_values = np.load('./data/shap_values.npy')

# Global SHAP on test
st.subheader("HGBM SHAP BARPLOT on test Values")
fig = plt.figure()
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns,plot_type='bar')
st.pyplot(fig)


