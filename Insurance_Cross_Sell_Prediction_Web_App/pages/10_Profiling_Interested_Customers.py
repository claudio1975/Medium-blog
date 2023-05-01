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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle5 as pickle

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


st.title("Profiling Interested Customers")

st.markdown("""
Interested Customers in buying a vehicle insurance coverage can be profiled into 4 clusters.
""")

df = pd.read_csv('./data/train.csv')

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')
df.drop(['id'], axis=1, inplace=True)

numerical_cols = [var for var in df.columns if df[var].dtype in ['float64','int64']]

df_2 = df.copy()

df_2 = df_2[df_2['Response']==1]

# Select numerical columns
num_1 = df_2[numerical_cols]

# Standardization of data
sc = StandardScaler()
num_sc = sc.fit_transform(num_1)

# loading in the model to predict on the data
with open('./data/kmeans.pkl', 'rb') as pickle_in:
    kmeans = pickle.load(pickle_in)

labels = kmeans.predict(num_sc)
cluster_num = num_1.copy()
cluster_num['kmeans_cluster'] = labels

df_cluster = pd.concat([df_2, cluster_num['kmeans_cluster']], axis=1)

st.subheader('Group clusters by Annual Premium')
# Group clusters by Annual Premium
df = df_cluster.groupby(df_cluster['kmeans_cluster'], as_index=False)['Annual_Premium'].sum()
df['PERCENTAGE'] = df['Annual_Premium']/df['Annual_Premium'].sum()*100
# dropping not matching rows
df = df.dropna()
# ranking 
df = df.sort_values(by = 'Annual_Premium', ascending = False).reset_index(drop=True)
df_AP = df.style.background_gradient(cmap='winter').format({'PERCENTAGE': "{:.2f}"}).format({'Annual_Premium':"{:,.2f}"})
df_AP

st.subheader('Group clusters by Age')
# Group clusters by Age
df = df_cluster.groupby(df_cluster['kmeans_cluster'], as_index=False)['Age'].mean()
df['PERCENTAGE'] = df['Age']/df['Age'].sum()*100
# dropping not matching rows
df = df.dropna()
# ranking 
df = df.sort_values(by = 'Age', ascending = False).reset_index(drop=True)
df_AGE = df.style.background_gradient(cmap='winter').format({'PERCENTAGE': "{:.2f}"}).format({'Age':"{:,.2f}"})
df_AGE

# Select top cluster per Annual Premium and Age
df_cluster_AP = df_cluster[df_cluster['kmeans_cluster']==2].reset_index(drop=True)
df_cluster_AGE = df_cluster[df_cluster['kmeans_cluster']==1].reset_index(drop=True)

st.markdown("""
In the following are showed the Annual Premium distribution and Age distribution for the top cluster.
""")

st.subheader('Annual Premium distribution')
# Plot Annual Premium vs Gender, Vehicle_Damage, Vehicle_Age, Previously_Insured
plt.rcParams['figure.figsize']=(15,12)
fig=plt.figure()
plt.subplot(2,2,1)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Gender'],palette="crest", multiple='stack')
plt.title('Annual_Premium vs Gender')
plt.subplot(2,2,2)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Vehicle_Damage'],palette="crest", multiple='stack')
plt.title('Annual_Premium vs Vehicle_Damage')
plt.subplot(2,2,3)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Vehicle_Age'],palette="crest", multiple='stack')
plt.title('Annual_Premium vs Vehicle_Age')
plt.subplot(2,2,4)
sns.kdeplot(x=df_cluster_AP['Annual_Premium'],hue=df_cluster_AP['Previously_Insured'],palette="crest", multiple='stack')
plt.title('Annual_Premium vs Previously_Insured')
st.pyplot(fig)

st.subheader('Age distribution')
# Plot Age vs Gender, Vehicle_Damage, Vehicle_Age, Previously_Insured
plt.rcParams['figure.figsize']=(15,12)
fig=plt.figure()
plt.subplot(2,2,1)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Gender'],palette="crest", multiple='stack')
plt.title('Age vs Gender on Top Cluster')
plt.subplot(2,2,2)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Vehicle_Damage'],palette="crest", multiple='stack')
plt.title('Age vs Vehicle_Damage on Top Cluster')
plt.subplot(2,2,3)
sns.kdeplot(x=df_cluster_AGE['Age'],hue=df_cluster_AGE['Vehicle_Age'],palette="crest", multiple='stack')
plt.title('Age vs Vehicle_Age on Top Cluster')
plt.subplot(2,2,4)
sns.kdeplot(x=df_cluster_AP['Age'],hue=df_cluster_AP['Previously_Insured'],palette="crest", multiple='stack')
plt.title('Age vs Previously_Insured')
st.pyplot(fig)

