#!/usr/bin/env python
# coding: utf-8

# ## Alternate Final Project for Robert Freid
# ### Original Final Project Failed
# I made the mistake of choosing data sets that were not suitable for the Final Project. <br>
# I am attemting to pull toether something that is more flexible and usable for the assignment requirments.

# In[202]:


# python 3.10 or greater is required
import sys
assert sys.version_info >= (3, 10)

# common imports
import numpy as np
import pandas as pd
import os

# Sickit Learn imports
## For pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

## For preprocessing
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.impute import(
    SimpleImputer
)

from sklearn.linear_model import (
    LogisticRegression
)

## For model selection
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV
)

# Classifier Algorithms
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from xgboost import XGBClassifier

# To plot figures
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
import plotly.offline as pyo
pyo.init_notebook_mode()
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
sns.set_style('whitegrid')

%matplotlib inline

# make notebook stable across runs
np.random.seed(42)

# In[203]:


dfloan = pd.read_csv("data/Loan_Data.csv")


# In[204]:


dfloan.shape


# In[205]:


# Display first five rows.
dfloan.head()


# In[206]:


# Check for null values
dfloan.isna().sum()


# In[207]:


# Check for duplicated rows.
dfloan.duplicated().sum()


# ## Checkpoint: Successfully loaded dataset, found missing or null values, no duplicated rows.

# In[208]:


dfloan.info()


# ## Summation: 8 catagorical columns, 5 numerical columns.

# In[209]:


# Examine the describe() function results.
dfloan.describe()


# In[210]:


# Examine the numerical dataset histograms for outliers and anomalies.
dfloan.hist(bins = 50, figsize=(20,20))


# ## Summation of histograms.
# 1.) Aplicant Income: Definitely has outliers and a majority of applicants income is centered around $5,000.00. <br>
# 2.) Coapplicant Income: Essential the same as applicant income but luch less, with fewer outliers. <br>
# 3.) Loan Amount: Loan amount just above 100 in number of applicants, tapers to the right. <br>
# 4.) Loan Amount Term: Appears to be in months. Maybe need to consider alternative way to represent this data. <br>
# 5.) Credit History: Appears to be either a 1 or 0, signifying that it is a simple has or has no credit history. <br>
# ## Cleaning the Data

# In[211]:


# Drop the Loan_ID column it provides no important information.
dfloan.drop(columns='Loan_ID', inplace=True)


# In[212]:


# Make sure the Loan_ID column has been droped from the datarframe.
dfloan.sample(5)


# 

# In[213]:


# Get the number of rows.
print('The total number of rows is: ', len(dfloan))


# In[214]:


# Get the total number of columns.
print('The total number of columns is: ', len(dfloan.columns))


# In[215]:


# Check again for the columns with null values.
dfloan.isna().sum()


# ## Display the value counts for each of the columns that have null or missing values.

# In[216]:


# Value_counts for the Gender column.
display(dfloan['Gender'].value_counts())


# ### Gender as a significant influencing catagory.
# There are only 13 out of 614 rows missing gender. <br>
# The ratio of men to women is 4-5 to 1. <br>
# There is little to no difference in credit scores between men and women. <br>
# Link: https://www.bankrate.com/personal-finance/debt/men-women-and-debt-does-gender-matter/

# In[217]:


# Fill Gender missing values with 'Female' value, not a significant edetermining element..
dfloan['Gender'].fillna('Female', inplace = True)


# In[218]:


# Value_counts for the Married column.
display(dfloan['Married'].value_counts())


# ### Married has only 3 missing values, and is not a significant influencing factor. 
# The ratio of married to not married is approximately 2 to 1. <br>
# Married people tend ot be better risks when loaning money than unmarried. <br>
# Unmarried people tend to be a little more slopy and might miss this category. <br>
# I am changing the null values to No for this reasons to preserve the other data.

# In[219]:


# Fill Married missing values with 'No' 
dfloan['Married'].fillna('No', inplace = True)


# In[220]:


# Value_counts for the Gender column.
display(dfloan['Dependents'].value_counts())


# ### Judging Dependents as factors in making Loans.
# There are 15 null values out of 614 rows. <br>
# Dependents above 3 is considered an increased risk in paying the loan back. <br>
# Link: https://www.nicheadvice.co.uk/how-children-affect-mortgage-applications/ <br>
# Therefore I am adding the 13 null values to the 3+ category attribute. <br>
# Individuals with more than 3 dependents would be most likely not to disclose that catagory. 

# In[221]:


# Fill Dependents missing values with '3+' 
dfloan['Dependents'].fillna('3+', inplace = True)


# In[222]:


# Value_counts for the Self_Employed column.
display(dfloan['Self_Employed'].value_counts())


# ### Self-Employed value counts.
# There are 32 out of 614 rows that are missing values for self employed individuals. <br>
# Self employed individuals are usually very proud and are happy to admit that they are  independent. <br>
# On the other hand, if someone has a bad credit history, they would not disclose that value. <br>
# I am chaning the null values for Self_Employed to 'No' to preserve the rest of the data.
# 

# In[223]:


# Fill Self_Employed missing values with 'No' 
dfloan['Self_Employed'].fillna('No', inplace = True)


# In[224]:


# Value_counts for the Loan Amount column.
display(dfloan['LoanAmount'].value_counts())


# ### Loan Amount has 22 missing values out of 614
# There are 203 different values for this catagory. <br>
# The loan amount could have a direct effect on the approval of the loan process. <br>
# And could impact the amount of risk the company is taking. <br>
# Best to eliminate these entries, which leave 592 records.

# In[225]:


# Drop row values where null based on the LoanAmount column.
dfloan.dropna(axis=0, subset=('LoanAmount'),inplace=True)


# In[226]:


# Display Loan_Amount_Term Value Counts
display(dfloan['Loan_Amount_Term'].value_counts())


# ### Loan Amount Term has 14 missing values out of the remaining 592 records
# The most prevalent value is 360, or 30 year loan. <br>
# Not a significant factor in the ability to repay the loan. <br>
# Will make null values equal to 360.

# In[227]:


# Fill Loan_Amount_Term missing values with '360' 
dfloan['Loan_Amount_Term'].fillna(360, inplace = True)


# In[228]:


# Display Credit_History Value Counts
display(dfloan['Credit_History'].value_counts())


# ### The credit History column has the most na values with there being 50, out of 592
# If we reaplce the null values with 0, it will force a review of the loan to acquire history. <br>
# Replace null values with 0.

# In[229]:


# Replace null values with 0.0
dfloan['Credit_History'].fillna(0.0, inplace = True)


# ## Summation of Null or Missing Values
# We should have 592 records. <br>
# We should have no null values.

# In[230]:


# Check again for the columns with null values.
dfloan.isna().sum()


# ## Review Data Set Types 
# Review the data set types and if necessary make changes to facilitate processing.

# In[231]:


dfloan.shape


# In[232]:


# Emaploy the info to gather numerical information.
dfloan.info()


# In[233]:


dfloan.sample(5)


# ## Analysis of Data Types
# Gender is either 'Male' or 'Female'. Needs to be changed to either 1 or 0. <br>
# Married is a yes or a no, change to true or false? <br>
# Dependents and Education is OK. <br>
# Self_Employed needs to be changed from a string (yes, no) to binary (true, false) value. <br>
# Credit_History needs to be changed from a numeric 1, 0  to binary true,false values. <br>
# Loan_Status needs to be changed fro a string Y, N to binary True, False values.

# In[234]:


# Change the Gender vales from Male, Female to 1, 0
dfloan['Gender'] = dfloan['Gender'].map({'Male': 1, 'Female': 0})


# In[235]:


# Change the Married column values from string, No and Yes to binary, true and false.
dfloan['Married'] = dfloan['Married'].map({'Yes':True, 'No': False})


# In[236]:


# Change the Self_Employed column values from string, No and Yes to binary, true and false.
dfloan['Self_Employed'] = dfloan['Self_Employed'].map({'Yes':True, 'No': False})


# In[237]:


# Change the Credit_Hsitory column values from numeric, 0, 1 to binary, true and false.
dfloan['Credit_History'] = dfloan['Credit_History'].map({ 1:True, 0:False})


# In[238]:


# Change the Loan_Status column values from string, Y, N to binary, true and false.
dfloan['Loan_Status'] = dfloan['Loan_Status'].map({ 'Y':True, 'N':False})


# In[239]:


dfloan.sample(5)


# In[240]:


dfloan.info()


# In[241]:


loan_corr = dfloan.corr()
loan_corr['Loan_Status'].sort_values(ascending=False)


# In[242]:


# Display Correlation Matrix for dataset
sns.heatmap(dfloan.corr(), cmap='BrBG');


# ## Spliting the Dataset.

# In[243]:


loan_X = dfloan.drop('Loan_Status',axis=1)
loan_y = dfloan['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(loan_X, loan_y, test_size=0.2, random_state=45)


# In[244]:


display(loan_X.sample(3))
display(loan_y.sample(3))


# ### Seperate the features from the labels.

# In[245]:


num_features = ['Gender', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'];
cat_features = ['Dependents', 'Education', 'Property_Area'];
col_selector = ['Married', 'Self_Employed', 'Credit_History'];

num_pipeline = Pipeline([
    ('scalar', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
    ('pass', "passthrough", col_selector)
])


# In[246]:


loan_prepared = full_pipeline.fit_transform(loan_X)

column_names = [
    feature
        .replace('num__', '')
        .replace('cat__', '')
        .replace('pass__', '')
    for feature in full_pipeline.get_feature_names_out()
]

loan_prepared = pd.DataFrame(loan_prepared, columns=column_names, index=loan_X.index)


# In[247]:


# display the loan_prepared dataset
loan_prepared.head()


# ## Analyze the Prepared Dataset

# In[248]:


dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(loan_prepared, loan_y)


# In[249]:


# Print the classifier score
dummy_classifier.score(loan_prepared, loan_y)


# In[250]:


# Use cross val score to get AUC score 
scores = cross_val_score(
    dummy_classifier, loan_X, loan_y,
    scoring = "roc_auc", cv=10
) 

print(f"Dummy Classifier  AUC: {scores.mean():.3f} STD: {scores.std():.2f}")


# ### Create Model and Execute
# We have 592 rows, far below the 100K recommended level by sickit-learn for algorithm choice. <br>
# The ensemble models for this project are: Random Forest, Logistic Regression, Gradient Boosting, ADA Boosts, XG Boost, Voting Classification <br>
# KFold parameters are: n_splits = 10, random_state = 42, shuffle = True <br>
# Check for errors related to the cross val score.

# In[251]:


loan_prepared.sample(5)


# In[252]:


for model in [
    RandomForestClassifier,
    LogisticRegression,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    XGBClassifier
]:
    classifier_model = model()
    kFold = KFold(
        n_splits=10, random_state=42, shuffle=True
    )
    
    scores = cross_val_score(
        classifier_model,
        loan_prepared,
        loan_y,
        scoring="roc_auc", cv=kFold
    )
    print(
        f"{model.__name__:22}  AUC: {scores.mean():.3f}  STD: {scores.std():.2f}"
    )


# In[253]:


# Use confusion matrix display to visualize for the best model
rf = RandomForestClassifier()
rf.fit(loan_prepared, loan_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=rf,
    X=loan_prepared, y=loan_y,
    cmap='Blues', colorbar=False
)
plt.show()


# In[254]:


# Use confusion matrix display to visualize for the best model
lg = LogisticRegression()
lg.fit(loan_prepared, loan_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=lg,
    X=loan_prepared, y=loan_y,
    cmap='Blues', colorbar=False
)
plt.show()


# In[255]:


# Use confusion matrix display to visualize for the best model
gb = GradientBoostingClassifier()
gb.fit(loan_prepared, loan_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=gb,
    X=loan_prepared, y=loan_y,
    cmap='Blues', colorbar=False
)
plt.show()


# In[256]:


# Use confusion matrix display to visualize for the best model
ada = AdaBoostClassifier()
ada.fit(loan_prepared, loan_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=ada,
    X=loan_prepared, y=loan_y,
    cmap='Blues', colorbar=False
)
plt.show()


# In[257]:


# Use confusion matrix display to visualize for the best model
xg = XGBClassifier()
xg.fit(loan_prepared, loan_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=xg,
    X=loan_prepared, y=loan_y,
    cmap='Blues', colorbar=False
)
plt.show()


# ### Random Forest Classifier is the Model that Performed WELL
# We will see if we can improve on the prediction using the GridSearchCV 

# In[258]:


# Employ these parameters to see if we can improve on the model and the accuracy.
param_grid = {
    'max_depth':[2],
    'random_state':[0]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    n_jobs=1,
    scoring="roc_auc"
).fit(loan_prepared,loan_y)
print(f'Best Score: {grid_search.best_score_}')
print(f'Best Params:  {grid_search.best_params_}')


# #### Very slight improvement using GridSearchCV.
# Without: AUC: 0.731  STD: 0.06 <br>
# With: 0.7566

# Check the X_test and y_test data sets for completeness

# In[259]:


print('The X_test dataset contains: ', X_test.sample(5))


# In[260]:


print('The y_test dataset contains:  ', y_test.sample(5))


# ### Test the test set for comparison to the training set.

# In[261]:


transformed_test_set = full_pipeline.transform(X_test)

column_names = [
    feature
        .replace('num__', '')
        .replace('cat__', '')
        .replace('pass__', '')
    for feature in full_pipeline.get_feature_names_out()
]

loan_test_prepared = pd.DataFrame(transformed_test_set, columns=column_names, index=X_test.index)
scores = cross_val_score(
    rf, loan_test_prepared, y_test,
    scoring = 'roc_auc', cv=10
)
print(
    f'Random Forest Classifier  AUC:  {scores.mean():.3f}  STD:  {scores.std():.2f}'
)


# #### Plot the Confusion Matrix for the best model for the test set.

# In[262]:


metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=rf,
    X=loan_test_prepared, y=y_test,
    cmap='Blues', colorbar=False
)
plt.show()


# # ***** End First Dataset - Begin Second Dataset *****

# In[263]:


# Load second loan dataset called datacreditos.csv for second data set for analysis.
dfloan2 = pd.read_csv('data/Credit_Data.csv')
dfloan2.head()


# In[264]:


# Drop the ID column as it is not relevant to the outcome.
dfloan2.drop(columns=['ID'], axis=1, inplace=True)
dfloan2.sample(5)


# In[265]:


# Check the shape.
dfloan2.shape


# In[266]:


# Check for null values.
dfloan2.isnull().sum()


# In[267]:


dfloan2.info()


# In[268]:


# Check the describe information.
dfloan2.describe()


# In[269]:


# Display in a histogrm information.
dfloan2.hist(bins=50, figsize=(20,20))


# In[270]:


# Valuecounts for columns of unknown row value variations.
display(dfloan2['Loan_Type'].value_counts())


# In[271]:


display(dfloan2['Gender'].value_counts())


# In[272]:


display(dfloan2['Degree'].value_counts())


# In[273]:


display(dfloan2['Citizenship'].value_counts())


# In[274]:


dfloan2.sample(5)


# ### Summation of Dataset Before Splitting Data Set into Train and Test
# No null values. <br>
# Nominal Categrical Columns: Loan_Type, Gender, Degree, Citizenship <br>
# Numerical Columns: Age, Income, Credit_score, Signers, and Default (Target Column) <br>
# Histograms show a very even distribution of the data. <br>
# ### Test Train Split Dataset

# In[275]:


train_set_l2, test_set_l2 = train_test_split(dfloan2, test_size=0.2, random_state=42)
train_set_l2.head()


# ### Process

# In[276]:


loan2_X = train_set_l2.drop('Default', axis=1)
loan2_y = train_set_l2['Default'].copy()


# In[277]:


loan2_X.head()


# In[278]:


loan2_y.head()


# In[279]:


from sklearn.compose import ColumnTransformer

num_features2 = ['Age', 'Income', 'Credit_score', 'Loan_length', 'Signers'];
cat_features2 = ['Loan_Type', 'Gender', 'Degree', 'Citizenship'];

num_pipeline2 = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline2 = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline2 = ColumnTransformer([
    ('num2', num_pipeline2, num_features2),
    ('cat2', cat_pipeline2, cat_features2)
])



# In[281]:


transformed_loan_set = full_pipeline2.fit_transform(loan2_X)

column_names=[
    feature 
        .replace('num2__', '')
        .replace('cat2__', '')
    for feature in full_pipeline2.get_feature_names_out()
]

loan2_prepared = pd.DataFrame(transformed_loan_set, columns=column_names, index=loan2_X.index)
loan2_prepared.head()


# ### Now we can Analyze our prepared data for best performance!

# In[282]:


for model in [
    RandomForestClassifier,
    LogisticRegression,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    XGBClassifier
]:
    classifier_model = model()
    kFold = KFold(
        n_splits=10, random_state=42, shuffle=True
    )
    
    scores = cross_val_score(
        classifier_model,
        loan2_prepared,
        loan2_y,
        scoring="roc_auc", cv=kFold
    )
    print(
        f"{model.__name__:22}  AUC: {scores.mean():.3f}  STD: {scores.std():.2f}"

    )


# In[283]:


lg = LogisticRegression()
lg.fit(loan2_prepared, loan2_y)

metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=lg,
    X=loan2_prepared, y=loan2_y,
    cmap='Blues', colorbar=False
)


# ### Use the GridSearchCV to try and get better results.

# In[285]:


# Employ these parameters to see if we can improve on the model and the accuracy.
param_grid = {
    'random_state':[0]
}
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    n_jobs=1,
    scoring="roc_auc"
).fit(loan2_prepared,loan2_y)
print(f'Best Score: {grid_search.best_score_}')
print(f'Best Params:  {grid_search.best_params_}')


# ### Analysis of results.
# Without GridSearchCV =   AUC: 0.794  STD: 0.05 <br>
# With GridSearchCV = 0.803 <br>
# ### Analysis of the Test Set

# In[288]:


loan2_X.head()


# In[290]:


loan2_y.head()


# In[291]:


loan2_test_prepared = full_pipeline2.fit_transform(loan2_X)

column_names=[
    feature 
        .replace('num2__', '')
        .replace('cat2__', '')
    for feature in full_pipeline2.get_feature_names_out()
]

loan2_test_prepared = pd.DataFrame(loan2_test_prepared, columns=column_names, index=loan2_X.index)
scores = cross_val_score(
    lg, loan2_test_prepared, loan2_y,
    scoring='roc_auc', cv=10
)
print(
    f'Logistic Regression  AUC:  {scores.mean():.3f}  STD: {scores.std():.2f}'
)


# ### Test Set is consistent with the Train Set
# Check out the Confusion Matrix Display for a good visual!

# In[292]:


metrics.ConfusionMatrixDisplay.from_estimator(
    estimator=lg,
    X=loan2_test_prepared, y=loan2_y,
    cmap='Blues', colorbar=False
)

