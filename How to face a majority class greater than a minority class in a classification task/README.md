In a classification task a typical problem is to face a majority class greater than a minority class, so the goal is to use 
techniques that allow to overcome this issue. 
For this purpose I've taken a data set from Kaggle to build a credit scoring model: https://www.kaggle.com/ajay1735/hmeq-data

The project is developed both in R & Python and in several steps.


R approach:

-Baseline models with Caret

-Sampling Methods with Caret (DOWN-SAMPLING, OVER-SAMPLING, SMOTE) with Caret

-Cost-Sensitive models with Caret

-Sampling Methods with H2O


Python approach:

-Baseline models (I've only encoded categorical variables and filled in missing values)

-Sampling Methods applied to the baseline models

-Stratified Cross-Validation applied to the baseline models

-Sampling Methods (OVER-SAMPLING, DOWN-SAMPLING) and Stratified Cross-Validation applied to the baseline models

-Sampling Methods (SMOTE, ADASYN) and Stratified Cross-Validation applied to the baseline models

