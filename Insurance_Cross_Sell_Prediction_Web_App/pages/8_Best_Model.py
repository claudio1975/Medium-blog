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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title("Best Model Prediction: Histogram-based Gradient Boosting Machine")

st.markdown("""
Histogram-based Gradient Boosting Machine shows the best performance, then it's been fine tuned both in the hyperparameters and threshold optimization,
and here the results: 
""")


DATA_URL = ('./data/train.csv')
df = pd.read_csv(DATA_URL)
DATA_URL_xtr = ('./data/X_train.csv')
X_train = pd.read_csv(DATA_URL_xtr)
DATA_URL_xte = ('./data/X_test.csv')
X_test = pd.read_csv(DATA_URL_xte)
DATA_URL_ytr = ('./data/y_train.csv')
y_train = pd.read_csv(DATA_URL_ytr)
DATA_URL_yte = ('./data/y_test.csv')
y_test = pd.read_csv(DATA_URL_yte)


# Model
# loading in the model to predict on the data
with open('./data/HGBM_tclassifier.pkl', 'rb') as pickle_in:
    HGBM_tclassifier = pickle.load(pickle_in)
    
# prediction
predictions_tr = HGBM_tclassifier.predict_proba(X_train)[:,1]
predictions_tr_ = pd.DataFrame(predictions_tr, columns=['Prediction'])
predictions_te = HGBM_tclassifier.predict_proba(X_test)[:,1]
predictions_te_ = pd.DataFrame(predictions_te, columns=['Prediction'])

# Evaluation
auc_train = roc_auc_score(y_train, predictions_tr)  
auc_test = roc_auc_score(y_test, predictions_te) 

# metrics table
d1 = {'evaluation': ['AUC'],
     'model': ['HGBM'],
    'train': [auc_train],
    'test': [auc_test]
        }
df1 = pd.DataFrame(data=d1, columns=['model','evaluation','train','test'])
print('HGBM evaluation on cross-sell prediction')
df1

# compute the tpr and fpr from the prediction
fpr, tpr, thresholds = roc_curve(y_test, predictions_te)

# Plot the ROC curve
plt.rcParams['figure.figsize']=(10,5)
fig = plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Adjust the threshold and compute the true positive rate (TPR) and false positive rate (FPR)
threshold = 0.21
y_pred = np.where(predictions_te >= threshold, 1, 0)
fpr_new, tpr_new, _ = roc_curve(y_test, y_pred)

# Plot the new point on the ROC curve
st.write('ROC on test')
plt.scatter(fpr_new, tpr_new, c='r', label='New Threshold = %0.2f' % threshold)
plt.legend(loc="lower right")
st.pyplot(fig)

# create a Pandas DataFrame
y_test_= np.array(y_test)
y_test_ = y_test_.flatten()
y_pred = y_pred.flatten()
df_2 = pd.DataFrame({'Actual': y_test_, 'Predicted': y_pred})
fig=plt.figure()
sns.countplot(x='value', hue='variable', data=pd.melt(df_2))
plt.title('True vs Predicted Labels')
st.pyplot(fig)



