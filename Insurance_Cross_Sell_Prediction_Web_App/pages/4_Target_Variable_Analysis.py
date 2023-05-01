import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('./data/train.csv')

# Formatting features
df['Driving_License'] = df['Driving_License'].astype('object')
df['Region_Code'] = df['Region_Code'].astype('object')
df['Previously_Insured'] = df['Previously_Insured'].astype('object')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('object')
df['Response'] = df['Response'].astype('object')

# Split data set between target variable and features
X_full = df.copy()
y = X_full.Response
X_full.drop(['Response'], axis=1, inplace=True)

st.title("Target Variable Analysis")


st.markdown("""
The outcome is a binary variable with 1 value for policyholders interested in purchasing the vehicle insurance, 
instead with 0 value for policyholders not interested in it.

Target variable shows imbalanced classes, where only 12% of policyholders would buy the vehicle coverage.
""")


def piechart(data, col1, col2):
    # Plot the target variable 
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.subplot(1,2,1)
    data.groupby(col1).count()[col2].plot(kind='pie',autopct='%.0f%%').set_title("Pie {} Variable Distribution".format(col1))
    plt.subplot(1,2,2)
    sns.countplot(x=data[col1], data=data).set_title("Barplot {} Variable Distribution".format(col1))
    st.pyplot(fig)


col1 = 'Response'
col2 = 'id'
piechart(df, col1, col2)


