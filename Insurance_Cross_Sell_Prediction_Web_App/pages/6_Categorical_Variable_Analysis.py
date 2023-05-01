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

st.title("Categorical Variable Analysis")

st.markdown("""

With data cleaning have been dropped two variables: 'Policy_Sales_Channel' and 'Region_Code' because they are not useful in the modelling activity,
given that data are allocated in many classes. In the first one variable, more or less 70% of data are covered by 3 channels, in the second variable 
one more or less 40% of data are allocated in 2 regions and the rest in other many other not relevant region codes.

Looking at the other variables, gender variable shows a prevalence of men policyholders: 54% male vs 46% female.

Almost all of policyholders have a driving license, and they own young vehicle: 53% of vehicles are in a range of 1-2 years.

Most of policyholders did not previously insured with the Company: 54% did not insured with Company vs 46% previusly insured.

In the last features vehicles with damage and without damage are equally distribuited in the portfolio.

""")

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [var for var in X_full.columns if
                    X_full[var].nunique() <= 15 and 
                    X_full[var].dtype == "object"]
# Subset with categorical features
cat = X_full[categorical_cols]

def piechart(data, col1, col2):
    # Plot the target variable 
    plt.rcParams['figure.figsize']=(15,5)
    fig = plt.figure()
    plt.subplot(1,2,1)
    data.groupby(col1).count()[col2].plot(kind='pie',autopct='%.0f%%').set_title("Pie {} Variable Distribution".format(col1))
    plt.subplot(1,2,2)
    sns.countplot(x=data[col1], data=data).set_title("Barplot {} Variable Distribution".format(col1))
    st.pyplot(fig)

piechart(df, col1='Gender', col2='id')
piechart(df, col1='Driving_License', col2='id')
piechart(df, col1='Previously_Insured', col2='id')
piechart(df, col1='Vehicle_Age', col2='id')
piechart(df, col1='Vehicle_Damage', col2='id')



