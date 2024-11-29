# %% [markdown]
# ## <p style="background-color:#fea162; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">DATA SCIENCE BorrowRisk PROJECT</p>

# %% [markdown]
# <a id="toc"></a>
# 
# ## <p style="background-color:#fea162; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Table of Contents</p>
# 
# - <a href="#0">Introduction</a><br>
# - <a href="#1">Downloading And Importing Modules, Loading Data & Data Review</a><br>
# - <a href="#2">Preprocessing</a><br>
#     - <a href="#2.1">Cleaning Data</a><br>
#     - <a href="#2.2">Handling Missing Values</a><br>
#     - <a href="#2.3">Outlier Analysis</a><br>
#     - <a href="#2.4">Strategies For Imbalanced Data</a><br>
#         - <a href="#2.4.1">SMOTE</a><br>
#         - <a href="#2.4.2">ADASYN</a><br>
#         - <a href="#2.4.3">Under Sampling</a><br>
# - <a href="#3">Exploratory Data Analysis (EDA)</a><br>
# - <a href="#4">Scaling, Categorical Variables, Splitting</a><br>
# - <a href="#5">Models</a><br>
#     - <a href="#5.1">KNN Classifier</a><br>
#     - <a href="#5.2">Logistic Regression</a><br>
#     - <a href="#5.3">Neural Networks</a><br>
#     - <a href="#5.4">Decision Tree</a><br>
#     - <a href="#5.5">Random Forest</a><br>
#     - <a href="#5.6">XGBoost</a><br>
#     - <a href="#5.7">LightGBM</a><br>
#     - <a href="#5.8">CatBoost</a><br>
# - <a href="#6">Conclusion</a>
# 

# %% [markdown]
# <a id="0"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ## Introduction
# 
# ### 1.1 Information About the Project
# This project is about creating a model that predicts if someone will default on their loan. It helps financial institutions lower risks and avoid losing money.
# 
# - **Objective:**  
#   The goal is to build a model that can classify whether a borrower will repay the loan or not. This is important for making better lending decisions.  
# 
# - **Scope:**  
#   The project will include data preparation, testing different machine learning models with different imbalance data strategy, and evaluating their accuracy. It won’t focus on deploying the model but will create a ready-to-use prototype.
# 
# ### 1.2 Description of the Dataset
# The dataset contains loan-related data about borrowers.
# 
# - **Source:** From Kaggle's Loan Default Dataset. [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data)
# 
# - **Size:** Before preprocessing : 148,671 rows and 34 columns.
#              After preprocessing : 148,671 rows and 10 columns
# - **Type:** Tabular data (a mix of numerical and categorical information).  
# 
# ### 1.3 Description of the Columns
# Here are the main columns in the dataset:
# 
# - **Target Variable:**  
#   - `Loan Default`: A binary value (1 if the borrower defaulted, 0 if they didn’t).  
# 
# - **Feature Variables:**  
# 
# 
# | **Column Name**           | **Description**                                                                                                     |
# |----------------------------|---------------------------------------------------------------------------------------------------------------------|
# | `ID`                      | Unique identifier for each loan record.                                                                             |
# | `year`                    | Year of loan issuance.                                                                                              |
# | `loan_limit`              | Loan limit category.                                                                                                |
# | `Gender`                  | Loan applicant gender.                                                                                              |
# | `approv_in_adv`           | Indicates whether the loan has been pre-approved.                                                                   |
# | `loan_type`               | Type of loan.                                                                                                       |
# | `loan_purpose`            | Purpose of the loan.                                                                                                |
# | `Credit_Worthiness`       | Credit reliability based on past repayment behavior.                                                                |
# | `open_credit`             | Indicates whether the applicant has other open credits.                                                             |
# | `business_or_commercial`  | Whether the loan is for business or commercial purposes.                                                             |
# | **`loan_amount`**         | **Total loan amount.**                                                                                              |
# | `rate_of_interest`        | Loan interest rate.                                                                                                 |
# | `Interest_rate_spread`    | Interest rate differential relative to a benchmark index.                                                            |
# | **`Upfront_charges`**     | **Initial charges for the loan.**                                                                                   |
# | **`term`**                | **Loan duration in months.**                                                                                        |
# | `Neg_ammortization`       | Indicates whether there is negative amortization.                                                                   |
# | `interest_only`           | Whether the loan allows for interest-only payments.                                                                 |
# | `lump_sum_payment`        | Indicates whether there is an option to pay in a single installment.                                                |
# | **`property_value`**      | **Value of the property associated with the loan.**                                                                 |
# | `construction_type`       | Type of construction of the property.                                                                               |
# | `occupancy_type`          | Type of property occupancy.                                                                                         |
# | `Secured_by`              | Type of loan guarantee.                                                                                             |
# | `total_units`             | Number of units related to the loan.                                                                                |
# | **`income`**              | **Loan applicant's income.**                                                                                        |
# | `credit_type`             | Type of credit check used.                                                                                          |
# | **`Credit_Score`**        | **Applicant's credit score.**                                                                                       |
# | `co-applicant_credit_type`| Type of credit check for co-applicants.                                                                              |
# | **`age`**                 | **Age range of the applicant.**                                                                                     |
# | `submission_of_application`| Loan application submission method.                                                                                 |
# | **`LTV` (Loan to Value)** | **Ratio of loan amount to property value.**                                                                          |
# | `Region`                  | Geographic region of the loan.                                                                                      |
# | `Security_Type`           | Type of loan security.                                                                                              |                                                           |
# | **`dtir1` (Debt to Income Ratio)** | **Debt/income ratio of the applicant.**                                                                     |
#  
# 
# 

# %% [markdown]
# <a id="1"></a>
# ## Downloading And Importing Modules, Loading Data & Data Review
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
if True:
    !pip install pandas==2.2.3
    !pip install numpy==1.26.4
    !pip install matplotlib==3.8.4
    !pip install seaborn==0.13.2
    !pip install missingno==0.5.2
    !pip install scikit-learn==1.5.2
    !pip install xgboost==2.1.2
    !pip install lightgbm==4.5.0
    !pip install catboost==1.2.7
    !pip install imbalanced-learn==0.12.4

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import missingno as msno
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report,roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
import random


# %%
df_main = pd.read_csv('Loan_Default.csv')
df = df_main.copy()
pd.set_option("display.max_columns", df.shape[-1]) 


# %%
df.sample(10)

# %%
df.info()

# %%
df.shape

# %%
df.describe(include='all')

# %%
df.isnull().sum()

# %% [markdown]
# <a id="2"></a>
# ## <p style="background-color:#fea162; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">Preprocessing</p>
# 
# 
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# <a id="2.1"></a>
# ## 2.1 Data Cleaning
# Cleaning the dataset is a critical step before analysis. Each column was carefully checked for common issues, such as:
# 
# - **Duplicates:** No duplicate records were found; all entries were unique.
# - **Inconsistent Formats:** The data was already consistent in formatting (e.g., date formats, string casing).
# - **Incorrect Data:** No obvious errors were detected, such as negative ages or future dates.
# 
# Since the dataset was clean, no additional cleaning steps were required.
# 
# ### Credit Factors
# According to [Forbes.com](https://www.forbes.com/advisor/personal-loans/personal-loan-requirements/), these credit factors are significant when evaluating loans:
# 
# - **Credit Score and History:** Reflects the borrower's reliability in repaying debt.
# - **Income:** Determines the borrower's ability to repay the loan.
# - **Debt-to-income Ratio:** The ratio of debt payments to income.
# - **Collateral:** Assets offered as security for the loan.
# - **Origination Fee:** A fee charged by the lender for processing the loan application.
# 

# %%
willremove = []

# %% [markdown]
# ### Data Cleaning: Column `ID`

# %%
df["ID"].sample(10)

# %%
df["ID"].nunique(),df.shape[0]

# %%
willremove.append("ID")

# %% [markdown]
# ### Data Cleaning: Column `year`

# %%
df["year"].sample(10)

# %%
df["year"].describe()

# %%
willremove.append("year")

# %% [markdown]
# ### Data Cleaning: Column `loan_limit`

# %%
df["loan_limit"].sample(10) 

# %%
df["loan_limit"].value_counts(dropna=False)

# %%
willremove.append("loan_limit")

# %% [markdown]
# ### Data Cleaning: Column `Gender`

# %%
df["Gender"].sample(10)

# %%
df["Gender"].value_counts(dropna=False) , df["Gender"].isnull().sum()

# %%
df[df["Gender"]=="Sex Not Available"].sample(3)

# %%
df.loc[df["Gender"] == "Sex Not Available", "Gender"] = np.nan

# %%
df["Gender"].value_counts(dropna=False), df["Gender"].isnull().sum()

# %%
willremove.append("Gender")

# %% [markdown]
# ### Data Cleaning: Column `approv_in_adv`

# %%
df["approv_in_adv"].sample(10)

# %%
df["approv_in_adv"].value_counts(dropna=False)

# %%
willremove.append("approv_in_adv")

# %% [markdown]
# ### Data Cleaning: Column `loan_type`

# %%
df["loan_type"].sample(10)

# %%
df["loan_type"].value_counts(dropna=False)

# %%
willremove.append("loan_type")

# %% [markdown]
# ### Data Cleaning: Column `loan_purpose`

# %%
df["loan_purpose"].sample(10)

# %%
df["loan_purpose"].value_counts(dropna=False)

# %%
willremove.append("loan_purpose")

# %% [markdown]
# ### Data Cleaning: Column `Credit_Worthiness`

# %%
df["Credit_Worthiness"].sample(10)

# %%
df["Credit_Worthiness"].value_counts(dropna=False)

# %%
willremove.append("Credit_Worthiness")

# %% [markdown]
# ### Data Cleaning: Column `open_credit`

# %%
df["open_credit"].sample(10)

# %%
df["open_credit"].value_counts(dropna=False)

# %%
willremove.append("open_credit")

# %% [markdown]
# ### Data Cleaning: Column `business_or_commercial`

# %%
df["business_or_commercial"].sample(10)

# %%
df["business_or_commercial"].value_counts(dropna=False)

# %%
willremove.append("business_or_commercial")

# %% [markdown]
# ### Data Cleaning: Column `loan_amount`

# %%
df["loan_amount"].sample(10)

# %%
df["loan_amount"].describe(),df["loan_amount"].isnull().sum()

# %% [markdown]
# ### Data Cleaning: Column `rate_of_interest`

# %%
df["rate_of_interest"].sample(10)

# %%
df["rate_of_interest"].describe(),df["rate_of_interest"].isnull().sum()

# %%
willremove.append("rate_of_interest")

# %% [markdown]
# ### Data Cleaning: Column `Interest_rate_spread`

# %%
df["Interest_rate_spread"].sample(10)

# %%
df["Interest_rate_spread"].describe(),df["Interest_rate_spread"].isnull().sum()

# %%
willremove.append("Interest_rate_spread")

# %% [markdown]
# ### Data Cleaning: Column `Upfront_charges`

# %%
df["Upfront_charges"].sample(10)

# %%
df["Upfront_charges"].describe(),df["Upfront_charges"].isnull().sum()

# %% [markdown]
# ### Data Cleaning: Column `term`

# %%
df["term"].sample(10)

# %%
df["term"].describe(),df["term"].isnull().sum()

# %%
df["term"].value_counts(dropna=False)

# %% [markdown]
# ### Data Cleaning: Column `Neg_ammortization`

# %%
df["Neg_ammortization"].sample(10)

# %%
df["Neg_ammortization"].value_counts(dropna=False)

# %%
willremove.append("Neg_ammortization")

# %% [markdown]
# ### Data Cleaning: Column `interest_only`

# %%
df["interest_only"].sample(10)

# %%
df["interest_only"].value_counts(dropna=False)

# %%
willremove.append("interest_only")

# %% [markdown]
# ### Data Cleaning: Column `lump_sum_payment`

# %%
df["lump_sum_payment"].sample(10)

# %%
df["lump_sum_payment"].value_counts(dropna=False)

# %%
willremove.append("lump_sum_payment")

# %% [markdown]
# ### Data Cleaning: Column `property_value`

# %%
df["property_value"].sample(10)

# %%
df["property_value"].describe(),df["property_value"].isnull().sum()

# %% [markdown]
# ### Data Cleaning: Column `construction_type`

# %%
df["construction_type"].sample(10)

# %%
df["construction_type"].value_counts(dropna=False)

# %%
willremove.append("construction_type")

# %% [markdown]
# ### Data Cleaning: Column `occupancy_type`

# %%
df["occupancy_type"].sample(10)

# %%
df["occupancy_type"].value_counts(dropna=False)

# %%
willremove.append("occupancy_type")

# %% [markdown]
# ### Data Cleaning: Column `Secured_by`

# %%
df["Secured_by"].sample(10)

# %%
df["Secured_by"].value_counts(dropna=False)

# %%
willremove.append("Secured_by")

# %% [markdown]
# ### Data Cleaning: Column `total_units`

# %%
df["total_units"].sample(10)    

# %%
df["total_units"].value_counts(dropna=False)

# %%
willremove.append("total_units")

# %% [markdown]
# ### Data Cleaning: Column `income`

# %%
df["income"].sample(10)

# %%
df["income"].describe(),df["income"].isnull().sum()

# %% [markdown]
# ### Data Cleaning: Column `credit_type`

# %%
df["credit_type"].sample(10)

# %%
df["credit_type"].value_counts(dropna=False)

# %%
willremove.append("credit_type")

# %% [markdown]
# ### Data Cleaning: Column `Credit_Score`

# %%
df["Credit_Score"].sample(10)

# %%
df["Credit_Score"].describe(),df["Credit_Score"].isnull().sum()

# %% [markdown]
# ### Data Cleaning: Column `co-applicant_credit_type`

# %%
df["co-applicant_credit_type"].sample(10)

# %%
df["co-applicant_credit_type"].value_counts(dropna=False)

# %%
willremove.append("co-applicant_credit_type")

# %% [markdown]
# ### Data Cleaning: Column `age`

# %%
df["age"].sample(10)

# %%
df["age"].value_counts(dropna=False)

# %% [markdown]
# ### Data Cleaning: Column `submission_of_application`

# %%
df["submission_of_application"].sample(10)

# %%
df["submission_of_application"].value_counts(dropna=False)

# %%
willremove.append("submission_of_application")

# %% [markdown]
# ### Data Cleaning: Column `LTV`

# %%
df["LTV"].sample(10)

# %%
df["LTV"].describe(),df["LTV"].isnull().sum()       

# %%


# %%
(df["LTV"] - 100*df["loan_amount"]/df["property_value"]).mean()

# %% [markdown]
# ### Data Cleaning: Column `Region`

# %%
df["Region"].sample(10)

# %%
df["Region"].value_counts(dropna=False)

# %%
willremove.append("Region")

# %% [markdown]
# ### Data Cleaning: Column `Security_Type`

# %%
df["Security_Type"].sample(10)

# %%
df["Security_Type"].value_counts(dropna=False)

# %%
willremove.append("Security_Type")

# %% [markdown]
# ### Data Cleaning: Column `Status`

# %%
df["Status"].sample(10)

# %%
df["Status"].value_counts(dropna=False) 

# %% [markdown]
# ### Data Cleaning: Column `dtir1`

# %%
df["dtir1"].sample(10)

# %%
df["dtir1"].describe(),df["dtir1"].isnull().sum()

# %%
(df["dtir1"] - df["loan_amount"]/df["income"])

# %%
df["dtir1"] = df["loan_amount"]/df["income"]

# %%
[i for i in df.columns if not i in willremove]

# %%
df_old = df.copy()

# %%
df.drop(willremove,axis=1,inplace=True)

# %%
df.columns

# %%
df.head()

# %% [markdown]
# <a id="2.2"></a>
# ## 2.2 Missing Value Analysis
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# Evaluate the dataset for missing values:
# 
# ### **Percentage of Missing Data**
# Below are the features with missing values and their corresponding percentages:
# 
# - **Upfront_charges:** 39,642 missing values (26.664% of the data).
# - **term:** 41 missing values (0.028% of the data).
# - **property_value:** 15,098 missing values (10.155% of the data).
# - **income:** 9,150 missing values (6.155% of the data).
# - **age:** 200 missing values (0.135% of the data).
# - **LTV:** 15,098 missing values (10.155% of the data).
# - **dtir1:** 24,121 missing values (16.225% of the data).
# 
# ### **Handling Missing Data**
# The following strategies were applied to handle missing values for each feature:
# 
# - **Upfront_charges:** filled in any missing `Upfront_charges` values with the average `Upfront_charges` of loans that have the same `loan_amount` and `term` combination.
# - **term:** filled in any missing `term` values with the mode `term` of loans that have the same `loan_amount`.
# - **property_value:** filled in any missing `property_value` values with the average `property_value` of loans that have the same `loan_amount`.
# - **income:** filled in any missing `income` values with the average `income` of loans that have the same `property_value`.
# - **age:** filled in any missing `age` values with the mode `age` of loans that have the same `property_value`.
# - **LTV:** filled in any missing `LTV` values with the mode `LTV` of loans that have the same `term`.
# - **dtir1:** filled in any missing `dtir1` values with the mode `dtir1` of loans that have the same `loan_amount`.
# 
# ### **Summary**
# The dataset contained several missing values across key features. Appropriate strategies were selected based on the nature of the data and the percentage of missing entries. By addressing these missing values, the dataset is now ready for further analysis and modeling.
# 

# %%
msno.matrix(df)

# %%
total = df.shape[0]
miss_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
miss_percent = {}
for col in miss_columns:
    null_count = df[col].isnull().sum()
    percent = (null_count/total) * 100
    miss_percent[col] = percent
    print("{} : {} ({}%)".format(col, null_count, round(percent, 3)))

# %%
df['dtir1'] = df['dtir1'].replace([np.inf, -np.inf], np.nan)

# %%
df.hist(bins = 50, figsize = (20, 15))
plt.show()

# %%
corr = df.drop(columns=["age"],axis=1).corr()
plt.figure(figsize=(20, 15))
heatmap = sb.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

# %%
df.isnull().sum()

# %%
df.dtypes

# %% [markdown]
# ### Handling Missing Values: Column `Upfront_charges`

# %%
uc_att_1 = df["Upfront_charges"].fillna(df.groupby('loan_amount')['Upfront_charges'].transform(lambda x: x.mean()))
uc_att_1.isnull().sum()

# %%
uc_att_2 = df["Upfront_charges"].fillna(df.groupby(['loan_amount','term'])['Upfront_charges'].transform(lambda x: x.mean()))
uc_att_2.isnull().sum()

# %%
np.max(uc_att_1 - uc_att_2), np.min(uc_att_1 - uc_att_2)

# %%
df["Upfront_charges"] = uc_att_2

# %% [markdown]
# ### Handling Missing Values: Column `term`

# %%
df["term"].fillna(df.groupby('loan_amount')['term'].transform(lambda x: x.mode()[0])).isnull().sum()

# %%
df["term"].fillna(df.groupby('loan_amount')['term'].transform(lambda x: x.mode()[0]),inplace=True)

# %% [markdown]
# ### Handling Missing Values: Column `property_value`

# %%
df["property_value"].fillna(df.groupby('loan_amount')['property_value'].transform(lambda x: x.mean())).isnull().sum()   

# %%
df["property_value"].fillna(df.groupby('loan_amount')['property_value'].transform(lambda x: x.mean()),inplace=True)   

# %% [markdown]
# ### Handling Missing Values: Column `income`

# %%
df["income"].fillna(df.groupby('property_value')['income'].transform(lambda x: x.mean())).isnull().sum()

# %%
df["income"].fillna(df.groupby('property_value')['income'].transform(lambda x: x.mean()),inplace=True)

# %%
df["income"]=df["income"].fillna(df["income"].mean())

# %%
df["income"].isnull().sum()

# %% [markdown]
# ### Handling Missing Values: Column `age`

# %%
df["age"].fillna(df.groupby('property_value')['age'].transform(lambda x: x.mode()[0])).isnull().sum()

# %%
df["age"].fillna(df.groupby('property_value')['age'].transform(lambda x: x.mode()[0]),inplace=True)

# %% [markdown]
# ### Handling Missing Values: Column `LTV`

# %%
(df["LTV"] - 100*df["loan_amount"]/df["property_value"]).mean()

# %%
df["LTV"]=100*df["loan_amount"]/df["property_value"]

# %%
df["LTV"].isnull().sum()

# %% [markdown]
# ### Handling Missing Values: Column `dtir1`

# %%
df["dtir1"].isnull().sum()

# %%
(df["loan_amount"]/df["income"]).isnull().sum()

# %%
df["dtir1"]=df["loan_amount"]/df["income"]

# %%
df['dtir1'] = df['dtir1'].replace([np.inf, -np.inf], np.nan)

# %%
df['dtir1'].isnull().sum()

# %%
df["dtir1"].fillna(df.groupby('loan_amount')['dtir1'].transform(lambda x: x.mean())).isnull().sum()

# %%
df["dtir1"].fillna(df.groupby('loan_amount')['dtir1'].transform(lambda x: x.mean()),inplace=True)

# %%
df = df.replace([np.inf, -np.inf], None)

# %%
df.isnull().sum()

# %% [markdown]
# ### Handling Missing Values: `Remaining Null Values`

# %%
df["Upfront_charges"]=df["Upfront_charges"].fillna(df["Upfront_charges"].mean())
df["property_value"]=df["property_value"].fillna(df["property_value"].mean())
df["LTV"]=df["LTV"].fillna(df["LTV"].mean())
df["dtir1"]=df["dtir1"].fillna(df["dtir1"].mean())

# %%
df.isnull().sum().sum()

# %%
del df_old
del df_main

cols = list(df.columns[:-2])
cols.extend([df.columns[-1],df.columns[-2]])

df = df[cols]
df.columns = df.columns.str.lower()
df_main = df.copy()

# %% [markdown]
# <a id="2.3"></a>
# ## 2.3 Outlier Analysis
# Identify and handle outliers in the data.Plot features using boxplots to visualize outliers.

# %%
numeric_cols = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(y=col, data=df)
    plt.title(f'Boxplot of {col}')  

plt.tight_layout()  
plt.show()

# %%
df.hist(bins = 50, figsize = (20, 15))
plt.show()

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(x='status', y=col, data=df)
    plt.title(f'Boxplot of {col} by Status')  

plt.tight_layout()  
plt.show()

# %% [markdown]
# <a id="2.4"></a>
# ## 2.4 Strategies for Handling Imbalanced Data
# <a id="2.4.1"></a>
# ### 2.4.1 SMOTE

# %%
df_2 = df.copy()
category_list = df["age"].astype('category').cat.categories.to_list()
df_2["age"] = df["age"].astype('category').cat.codes

# %%
category_list

# %%
y = df_2["status"].copy()
X = df_2.drop("status", axis=1).copy()
del df_2

# %%
smote = SMOTE(random_state=1)
X_smote, y_smote = smote.fit_resample(X, y)
df_smote = pd.concat([X_smote, y_smote], axis=1)
np.bincount(y_smote)

# %%
df_smote.head()

# %%
X_smote["age"].value_counts()

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_smote.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(y=col, data=df_smote)
    plt.title(f'Boxplot of {col}')  

plt.tight_layout()  
plt.show()

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_smote.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(x='status', y=col, data=df_smote)
    plt.title(f'Boxplot of {col} by Status')  

plt.tight_layout()  
plt.show()

# %% [markdown]
# <a id="2.4.2"></a>
# ### 2.4.2 ADASYN

# %%
adasyn = ADASYN(random_state=1)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
df_adasyn = pd.concat([X_adasyn, y_adasyn], axis=1)
np.bincount(y_adasyn)

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_adasyn.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(y=col, data=df_adasyn)
    plt.title(f'Boxplot of {col}')  

plt.tight_layout()  
plt.show()

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_smote.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(x='status', y=col, data=df_smote)
    plt.title(f'Boxplot of {col} by Status')  

plt.tight_layout()  
plt.show()

# %%
X_adasyn["age"].value_counts()

# %% [markdown]
# <a id="2.4.3"></a>
# ### 2.4.3 Under Sampling

# %%
under_sampler = RandomUnderSampler(random_state=42)
X_under, y_under = under_sampler.fit_resample(X, y)
df_under = pd.concat([X_under, y_under], axis=1)
np.bincount(y_under)

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_smote.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(y=col, data=df_smote)
    plt.title(f'Boxplot of {col}')  

plt.tight_layout()  
plt.show()

# %%
plt.figure(figsize=(18, 12))  
sb.set_theme(style='darkgrid')

for i, col in enumerate(df_smote.columns[:-1]):
    plt.subplot(3, 3, i + 1)  
    sb.boxplot(x='status', y=col, data=df_smote)
    plt.title(f'Boxplot of {col} by Status')  

plt.tight_layout()  
plt.show()

# %%


# %% [markdown]
# <a id="3"></a>
# ## Exploratory Data Analysis (EDA)
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ## 3.1 Data Visualization

# %%
plt.figure(figsize=(20, 60))  
sb.set_theme(style='darkgrid')

df_sets = [df, df_smote, df_adasyn, df_under]
methods_for_imbalanced_data = ['No Imbalance Strategy', 'SMOTE Data', 'ADASYN Data', 'Under Sampled Data']

for i,col in enumerate(df.columns[:-1]):
    for j,title in enumerate(methods_for_imbalanced_data):
        plt.subplot(10, 4, i*4 + j+1)  
        sb.histplot(data=df_sets[j], x=col, hue="status", multiple="dodge", shrink=.8, bins=4)
        plt.title('Boxplot of {} by Status {}'.format(col,title))  
        

plt.tight_layout()  
plt.show()

# %%
df.hist(bins = 50, figsize = (20, 15))
plt.show()

# %%
df_adasyn.hist(bins = 50, figsize = (20, 15))
plt.show()

# %%
df_smote.hist(bins = 50, figsize = (20, 15))
plt.show()

# %%
df_under.hist(bins = 50, figsize = (20, 15))
plt.show()

# %% [markdown]
# ## 3.2 Correlation Analysis
# Analyze correlations between numerical features

# %%
for i, method in enumerate(methods_for_imbalanced_data):
    corr = df_sets[i%4].drop(columns=["age"],axis=1).corr()
    plt.figure(figsize=(20, 15))
    heatmap = sb.heatmap(corr, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap of {}'.format(method), fontdict={'fontsize':12}, pad=12)

# %%
for i, method in enumerate(methods_for_imbalanced_data):
    the_set = df_sets[i%4].drop(columns=["age"],axis=1)
    the_set["combined_loan_property"] = the_set["loan_amount"]*the_set["property_value"]
    corr = the_set.drop(columns=["loan_amount","property_value"],axis=1).corr()
    plt.figure(figsize=(20, 15))
    heatmap = sb.heatmap(corr, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap of {}'.format(method), fontdict={'fontsize':12}, pad=12)

# %%
X["combined_loan_property"] = X["loan_amount"]*X["property_value"]
X.drop(columns=["loan_amount","property_value"],axis=1,inplace=True)

# %%
X.columns

# %% [markdown]
# <a id="4"></a>
# ## 4. Scaling, Categorical Variables, and Splitting
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# ## 4.1 Scaling
# Both **Standard Scaler** and **Min-Max Scaler** were applied to the dataset for scaling numerical features. 
# 
# - **Standard Scaler:** Scales the data to have a mean of 0 and a standard deviation of 1, which is useful for algorithms sensitive to feature scaling (e.g., Logistic Regression).
# - **Min-Max Scaler:** Transforms the data to a fixed range (e.g., 0 to 1), which is beneficial for algorithms that rely on normalized inputs (e.g., Neural Networks).
# 
# By applying both scalers, the dataset can be tested with various models to determine the optimal scaling method for performance.

# %%
scalling_methods = ["No Scaling", "Standard Scaler", "Min Max Scaler"]
all_types_data = {i: {j : {k : None for k in ['X_train','X_test','y_train','y_test']}
                      for j in scalling_methods}
                      for i in methods_for_imbalanced_data}

# %%
age_column = X.pop('age')

# %%
X.dtypes

# %%
X = X.astype('float64')
X.dtypes

# %%
minmax = MinMaxScaler()
standard = StandardScaler()
X_minmax = X.copy()
X_standard = X.copy()
X_minmax[:] = minmax.fit_transform(X[:])
X_standard[:] = standard.fit_transform(X[:])


# %% [markdown]
# ## 4.2 Encoding Categorical Variables
# Only one feature in the dataset requires encoding.

# %%
X = pd.concat([X,pd.get_dummies(age_column,
                         drop_first=False,dtype=float,prefix='age')],axis=1)

X_minmax = pd.concat([X_minmax,pd.get_dummies(age_column,
                         drop_first=False,dtype=float,prefix='age')], axis=1)

X_standard = pd.concat([X_standard,pd.get_dummies(age_column,
                        drop_first=False,dtype=float,prefix='age')], axis=1)

# %%
X.head(1)

# %%
X_minmax.head(1)

# %%

Xs = [X,X_smote,X_adasyn,X_under]
ys = [y,y_smote,y_adasyn,y_under]
for index,data_frame in enumerate(methods_for_imbalanced_data):
    X_train, X_test, y_train, y_test = train_test_split(Xs[index], ys[index], test_size=0.3, stratify=ys[index], random_state=1)
    all_types_data[data_frame]["No Scaling"]["X_train"]=X_train
    all_types_data[data_frame]["No Scaling"]["X_test"]=X_test
    for method in scalling_methods:
        all_types_data[data_frame][method]["y_train"]=y_train
        all_types_data[data_frame][method]["y_test"]=y_test

# %%
X_standard.head(1)

# %% [markdown]
# ### 4.3 Splitting the Data into Training and Testing Sets
# 
# To prevent data leakage and ensure robust evaluation, the dataset was split into training and testing subsets as follows:
# 
# - **Train/Test Split:** The data was split into 80% for training and 20% for testing.
# - **Stratified Sampling:** Since the dataset is imbalanced, stratified sampling was applied to maintain the proportion of classes in both training and testing sets, ensuring that the minority class is adequately represented in each subset.

# %%
train_index, test_index = train_test_split(X.index, test_size=0.3, stratify=y, random_state=1)

# %%
def imbalance_strategy_applier(imbalance_method,data_X,data_y):
    if imbalance_method == "No Imbalance Strategy":
        return data_X,data_y
    elif imbalance_method == 'SMOTE Data':
        smote = SMOTE(random_state=1)
        X_smote, y_smote = smote.fit_resample(data_X, data_y)
        return X_smote, y_smote
    elif imbalance_method == 'ADASYN Data':
        adasyn = ADASYN(random_state=1)
        X_adasyn, y_adasyn = adasyn.fit_resample(data_X, data_y)
        return X_adasyn, y_adasyn
    elif imbalance_method == 'Under Sampled Data':
        under_sampler = RandomUnderSampler(random_state=1)
        X_under, y_under = under_sampler.fit_resample(data_X, data_y)
        return X_under, y_under


# %%
methods_for_imbalanced_data = ['No Imbalance Strategy', 'SMOTE Data', 'ADASYN Data', 'Under Sampled Data']
for imbalance_method in methods_for_imbalanced_data:
    for method_name, data in zip(scalling_methods, [X, X_standard, X_minmax]):
        data_X, data_y = imbalance_strategy_applier(imbalance_method,
                                                    data.loc[train_index],
                                                    y[train_index])
        all_types_data[imbalance_method][method_name]["X_train"] = data_X 
        all_types_data[imbalance_method][method_name]["y_train"] = data_y
        all_types_data[imbalance_method][method_name]["X_test"] = data.loc[test_index]
        all_types_data[imbalance_method][method_name]["y_test"] = y.loc[test_index]

# %% [markdown]
# <a id="5"></a>
# ## 5. Models
# - <a href="#5.1">KNN Classifier</a><br>
# - <a href="#5.2">Logistic Regression</a><br>
# - <a href="#5.3">Neural Networks</a><br>
# - <a href="#5.4">Decision Tree</a><br>
# - <a href="#5.5">Random Forest</a><br>
# - <a href="#5.6">XGBoost</a><br>
# - <a href="#5.7">LightGBM</a><br>
# - <a href="#5.8">CatBoost</a><br>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ## 5.1 Creating Models and Fine-Tuning
# 

# %%
def visualize_model_performance(results, model_name):
    # Prepare data for visualization
    performance_data = []
    for data_frame in results[model_name]:
        for scaling_method in results[model_name][data_frame]:
            test_results = results[model_name][data_frame][scaling_method].get('Test', {})
            performance_data.append({
                'Imbalance Method': data_frame,
                'Scaling Method': scaling_method,
                'Accuracy': test_results.get('Accuracy', np.nan),
                'F1 Score': test_results.get('F1 Score', np.nan),
                'Recall': test_results.get('Recall', np.nan),
                'ROC AUC': test_results.get('ROC', np.nan)
            })
    
    df_performance = pd.DataFrame(performance_data)
    # Set up the matplotlib figure
    plt.figure(figsize=(20, 15))

    metrics = ['Accuracy', 'F1 Score', 'Recall', 'ROC AUC']
    # Heatmaps for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        
        pivot_metric = df_performance.pivot(
            index='Imbalance Method', 
            columns='Scaling Method', 
            values=metric
        )
        sb.heatmap(pivot_metric, annot=True, cmap='YlGnBu', center=pivot_metric.mean().mean())
        plt.title(f'{metric} Heatmap')
        plt.tight_layout()
    
    # 5. Comparative Bar Plot (Recall)
    ax = plt.subplot(2, 3, 5)
    sb.barplot(
        x='Imbalance Method', 
        y='Recall', 
        hue='Scaling Method', 
        data=df_performance
    )
    ax.set_ylim([0, 1])
    plt.title(f'{model_name} Recall Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 6. Normalized Performance Radar Chart
    ax = plt.subplot(2, 3, 6)
    ax.set_ylim([0, 1])
    df_melted = pd.melt(
        df_performance, 
        id_vars=['Imbalance Method', 'Scaling Method'], 
        value_vars=['Accuracy', 'F1 Score', 'ROC AUC'],
        var_name='Metric', 
        value_name='Value'
    )
    
    sb.boxplot(
        x='Imbalance Method', 
        y='Value', 
        hue='Metric', 
        data=df_melted
    )
    plt.title('Overall Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Adjust layout and show plot
    plt.suptitle(f'{model_name} Model Performance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print detailed performance summary
    print(f"\n{model_name} Performance Summary:")
    summary = df_performance.groupby(['Imbalance Method', 'Scaling Method']).mean()
    print(summary)
    
    # Identify best configurations
    print(f"\nBest {model_name} Configurations:")
    for metric in metrics:
        best_idx = df_performance[metric].idxmax()
        print(f"\nBest {metric} Configuration:")
        print(df_performance.loc[best_idx])
    
    return None #df_performance

def set_save_results(model_list, model_name, data_frame, method, train_accuracy,test_metric, model, results):
    results[model_name][data_frame][method]["Train"] = train_accuracy
    results[model_name][data_frame][method]["Test"] = test_metric
    model_list[data_frame+"_"+method] = model

def run_model(model, all_types_data, data_frame,method, importance=False):
    model.fit(all_types_data[data_frame][method]["X_train"], all_types_data[data_frame][method]["y_train"])
    prediction = model.predict(all_types_data[data_frame][method]["X_test"])
    train_accuracy = model.score(all_types_data[data_frame][method]["X_train"], 
                                                                all_types_data[data_frame][method]["y_train"])
    test_metric = {
                "Accuracy": accuracy_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Precision": precision_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Recall": recall_score(all_types_data[data_frame][method]["y_test"], prediction),
                "F1 Score": f1_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Confusion Matrix": confusion_matrix(all_types_data[data_frame][method]["y_test"], prediction),
                "ROC": roc_auc_score(all_types_data[data_frame][method]["y_test"], prediction)
            }
    print("Train Score : ",train_accuracy)
    print("Test Score : ",test_metric["Accuracy"])
    print("Test F1 Score : ",test_metric["F1 Score"])
    print("Test ROC : ", test_metric["ROC"])
    print("Test Recall : ", test_metric["Recall"])
    if importance:
        print("Feature Importance : ",model.feature_importances_)
    return train_accuracy,test_metric, model

def run_check_model(model, results, all_types_data, data_frame,method, model_name, importance=False):
    model.fit(all_types_data[data_frame][method]["X_train"], all_types_data[data_frame][method]["y_train"])
    prediction = model.predict(all_types_data[data_frame][method]["X_test"])
    train_accuracy = model.score(all_types_data[data_frame][method]["X_train"], 
                                                                all_types_data[data_frame][method]["y_train"])
    test_metric = {
                "Accuracy": accuracy_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Precision": precision_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Recall": recall_score(all_types_data[data_frame][method]["y_test"], prediction),
                "F1 Score": f1_score(all_types_data[data_frame][method]["y_test"], prediction),
                "Confusion Matrix": confusion_matrix(all_types_data[data_frame][method]["y_test"], prediction),
                "ROC": roc_auc_score(all_types_data[data_frame][method]["y_test"], prediction)
            }
    print("Train Score : ",train_accuracy - results[model_name][data_frame][method]["Train"])
    print("Test Score : ",test_metric["Accuracy"] - results[model_name][data_frame][method]["Test"]["Accuracy"])
    print("Test F1 Score : ", test_metric["F1 Score"]- results[model_name][data_frame][method]["Test"]["F1 Score"])
    print("Test ROC : ", test_metric["ROC"] - results[model_name][data_frame][method]["Test"]["ROC"])
    print("Test Recall : ", test_metric["Recall"] - results[model_name][data_frame][method]["Test"]["Recall"])
    if importance:
        print("Feature Importance : ",model.feature_importances_)
    return train_accuracy,test_metric, model

def plot_model_comparison(y_true, y_pred_model1, y_pred_model2, model1_name="Model 1", model2_name="Model 2"):

    # Calculate metrics for both models
    metrics_model1 = {
        'Accuracy': accuracy_score(y_true, y_pred_model1),
        'Precision': precision_score(y_true, y_pred_model1),
        'Recall': recall_score(y_true, y_pred_model1),
        'F1 Score': f1_score(y_true, y_pred_model1)
    }
    
    metrics_model2 = {
        'Accuracy': accuracy_score(y_true, y_pred_model2),
        'Precision': precision_score(y_true, y_pred_model2),
        'Recall': recall_score(y_true, y_pred_model2),
        'F1 Score': f1_score(y_true, y_pred_model2)
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Bar plot comparing metrics
    plt.subplot(2, 1, 1)
    x = np.arange(len(metrics_model1))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, metrics_model1.values(), width, label=model1_name)
    bars2 = plt.bar(x + width/2, metrics_model2.values(), width, label=model2_name)
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics_model1.keys())
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    plt.grid(True, alpha=0.3)
    
    # 2. Heatmap comparison
    plt.subplot(2, 1, 2)
    metrics_array = np.array([
        list(metrics_model1.values()),
        list(metrics_model2.values())
    ])
    
    sb.heatmap(metrics_array, 
                annot=True, 
                fmt='.3f',
                xticklabels=list(metrics_model1.keys()),
                yticklabels=[model1_name, model2_name],
                cmap='YlOrRd')
    plt.title('Metrics Comparison Heatmap')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


# %%
model_names = ["KNN","Decision Tree","Random Forest","Logistic Regression","MLP","XGBoost","LightGBM","CatBoost"]
results = {l:{i:{j:{k:None for k in ["Test","Train"]} for j in scalling_methods} for i in methods_for_imbalanced_data} for l in model_names}

# %% [markdown]
# <a id="5.1"></a>
# ## 5.1 KNN Classifier
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
KNN = {}
model_number = 0
c=0
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Model: {}".format(model_names[model_number]))
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = KNeighborsClassifier(n_neighbors=5, # 5, 7, 10, 15, 20, 25
                                     algorithm='auto', # auto, ball_tree, kd_tree, brute
                                     leaf_size=30, # 30, 40, 50, 60, 70, 80, 90, 100
                                     p=2) # 1, 2
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(KNN, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")
        

# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "ADASYN Data"
method = "Standard Scaler"
model_number = 0
model = KNeighborsClassifier(n_neighbors=3, # 5, 7, 10, 15, 20, 25
                            weights='uniform', # 
                            algorithm='auto', 
                            leaf_size=30, # 30, 40, 50, 60, 70, 80, 90, 100
                            p=2 # 1, 2
                            ) 

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method,model_names[model_number], importance=False)

# %%
# set_save_results(KNN, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.2"></a>
# ## 5.2 Logistic Regression
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
LR = {}
c=0
model_number = 3
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = LogisticRegression(class_weight="balanced", 
                                   solver='lbfgs', 
                                   max_iter=400,  # 100, 200, 300, 400, 500
                                   warm_start=True)
        
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(LR, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")


# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "SMOTE Data"
method = "Min Max Scaler"
model_number = 3

model = LogisticRegression(penalty='l2', # 'l1', 'l2', 'elasticnet
                            dual=False,
                            tol=1e-4,
                            C=1.0,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight="balanced", # 'balanced', None
                            solver='liblinear', # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
                            max_iter=100,
                            multi_class='auto',
                            verbose=0,
                            warm_start=False)

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method,model_names[model_number], importance=False)


# %%
# set_save_results(LR, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.3"></a>
# ## 5.3 Neural Networks 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
MLP = {}
model_number = 4
c=0
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = MLPClassifier(hidden_layer_sizes=(64, ), # (64,32 ), (32, 16), (128, 64, 32)
                               activation='relu', 
                                 solver='adam', # 'lbfgs', 'sgd', 'adam'
                                alpha=0.0001,
                                batch_size='auto',
                                learning_rate='constant', # 'constant', 'invscaling', 'adaptive'
                                learning_rate_init=0.001,
                                power_t=0.5,
                                max_iter=200,
                                shuffle=True,
                                random_state=42,
                                n_iter_no_change=10)
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(MLP, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")


# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "ADASYN Data"
method = "Standard Scaler"
model_number = 4
model = MLPClassifier(hidden_layer_sizes=(64,32,16 ), # (64,32 ), (32, 16), (128, 64, 32)
                                 activation='relu', #
                                    solver='sgd', # 'lbfgs', 'sgd', 'adam'
                                  alpha=0.0001,
                                  batch_size='auto',
                                  learning_rate='adaptive', # 'constant', 'invscaling', 'adaptive'
                                  learning_rate_init=0.001,
                                  power_t=0.5,
                                  max_iter=200,
                                  shuffle=True,
                                  random_state=42,
                                  tol=0.0001,
                                  verbose=False,
                                  warm_start=False,
                                  momentum=0.9,
                                  nesterovs_momentum=True,
                                  early_stopping=False,
                                  validation_fraction=0.1,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-08,
                                  n_iter_no_change=10,
                                  max_fun=15000)

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=False)


# %%
# set_save_results(MLP, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.4"></a>
# ## 5.4 Decision Tree  
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
Decision_Tree = {}
c=0
model_number = 1
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = DecisionTreeClassifier(criterion='gini', # 'gini', 'entropy'
                                       min_samples_split=2, 
                                       min_samples_leaf=1, 
                                       random_state=43,  
                                       class_weight="balanced")
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(Decision_Tree, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")

# %%
visualize_model_performance(results,model_name=model_names[model_number])

# %%
data_frame = "Under Sampled Data"
method = "Min Max Scaler"
model_number = 1
model = DecisionTreeClassifier(criterion='entropy', 
                                splitter='best', 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                random_state=10, 
                                min_impurity_decrease=0.0, 
                                class_weight="balanced")

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=True)

# %%
# set_save_results(Decision_Tree, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.5"></a>
# ## 5.5 Random Forest  
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
RF = {}
c= 0
model_number = 2
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = RandomForestClassifier(n_estimators=200, 
                                       criterion='entropy', #gini
                                       min_samples_split=2, 
                                       min_samples_leaf=1, 
                                       min_weight_fraction_leaf=0.0,  
                                       min_impurity_decrease=0.0, 
                                       bootstrap=True, 
                                       random_state=42, 
                                       verbose=0, 
                                       warm_start=False, 
                                       class_weight="balanced", 
                                       ccp_alpha=0.0)
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(RF, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")

# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "No Imbalance Strategy"
method = "Standard Scaler"
model_number = 2
model = RandomForestClassifier(n_estimators=300, 
                            criterion='entropy', 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            min_impurity_decrease=0.0, 
                            bootstrap=False, # True, False
                            max_features=6, # 6, 8, 10, 12, 14, 16
                            oob_score=False, 
                            random_state=1, 
                            warm_start=False, 
                            class_weight="balanced")

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=True)


# %%
# set_save_results(RF, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.6"></a>
# ## 5.6 XGBoost   
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
XGB = {}
c=0
model_number = 5
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Progress: {:.2f}%".format(100*(c)/(len(all_types_data)*len(scalling_methods))))
        c+=1
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        class_counts = all_types_data[data_frame][method]["y_train"].value_counts()
        num_negative = class_counts[0]
        num_positive = class_counts[1]
        scale_pos_weight = num_negative / num_positive
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, 
                              n_estimators=100, 
                              max_depth=3)
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(XGB, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")

# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "ADASYN Data"
method = "No Scaling"
class_counts = all_types_data[data_frame][method]["y_train"].value_counts()
num_negative = class_counts[0]
num_positive = class_counts[1]
scale_pos_weight = num_negative / num_positive
model_number = 5
model = XGBClassifier(learning_rate=0.01, 
                              n_estimators=300, 
                              max_depth=5, 
                              min_child_weight=1, 
                              gamma=1, 
                              subsample=1, 
                              colsample_bytree=1, 
                              objective='binary:logistic', 
                              nthread=1, 
                              scale_pos_weight=scale_pos_weight, 
                              seed=27, 
                              use_label_encoder=False)

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=True)


# %%
# set_save_results(XGB, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.7"></a>
# ## 5.7 LightGBM   
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
LGBM = {}
model_number = 6
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Model: {}".format(model_names[model_number]))
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = LGBMClassifier(boosting_type='gbdt', 
                              class_weight='balanced')
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(LGBM, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")

# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "SMOTE Data"
method = "No Scaling"
model_number = 6
model = LGBMClassifier(boosting_type='gbdt',
                        class_weight='balanced',
                        objective='binary',
                        random_state=42,
                        n_estimators=200,
                        learning_rate=0.01,
                        max_depth=4)

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=True)

# %%
# set_save_results(LGBM, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# <a id="5.8"></a>
# ## 5.8 CatBoost   
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>

# %%
CatBoost = {}
model_number = 7
for data_frame in all_types_data:
    for method in scalling_methods:
        print("Model: {}".format(model_names[model_number]))
        print("Data Frame : {} , Scalling Method : {}".format(data_frame,method))
        model = CatBoostClassifier(iterations=100,
                              learning_rate=0.01,
                              depth=6,
                              loss_function='Logloss',
                              verbose=True)
        train_accuracy,test_metric, model = run_model(model, all_types_data, data_frame,method, importance=False)
        set_save_results(CatBoost, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)
        print("---------------------------------------------------")

# %%
visualize_model_performance(results,model_names[model_number])

# %%
data_frame = "No Imbalance Strategy"
method = "No Scaling"
model_number = 7
model = CatBoostClassifier(iterations=1000,
                              learning_rate=0.001,
                              depth=9,
                              loss_function='Logloss',
                              verbose=True)

train_accuracy,test_metric, model = run_check_model(model, results, all_types_data, data_frame,method, model_names[model_number], importance=True)

# %%
# set_save_results(CatBoost, model_names[model_number], data_frame, method, train_accuracy,test_metric, model, results)

# %% [markdown]
# ## 5.2 Model Comparisons
# Compare the performance of different models.

# %%
results_backup = results.copy()

# %%
train_accuracies = []
test_accuracies = []
test_recalls = []
test_precisions = []
test_f1s = []
test_ROCs = []
x_labels = []

for model_name, models in results.items():
    for preprocess, imbalanced_strategies in models.items():
        for scaling, metrics in imbalanced_strategies.items():
                train_accuracy = metrics.get("Train", None)
                test_metrics = metrics.get("Test", None)
                test_accuracy = test_metrics.get("Accuracy", None) 
                test_recall = test_metrics.get("Recall", None) 
                test_precision = test_metrics.get("Precision", None) 
                test_f1 = test_metrics.get("F1 Score", None) 
                test_ROC = test_metrics.get("ROC", None) 

                if train_accuracy is not None and test_accuracy is not None:
                    x_labels.append(f"{model_name}-{preprocess}({scaling})")
                    train_accuracies.append(train_accuracy)
                    test_accuracies.append(test_accuracy)
                    test_recalls.append(test_recall)
                    test_ROCs.append(test_ROC)
                    test_precisions.append(test_precision)
                    test_f1s.append(test_f1)
                
               


data = list(zip(x_labels, train_accuracies, test_accuracies, test_recalls))
data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
x_labels_sorted, train_accuracies_sorted, test_accuracies_sorted, test_recalls_sorted = zip(*data_sorted)

# Plotting
x = range(len(x_labels_sorted)) 
width = 0.35 

plt.figure(figsize=(40, 12))

plt.bar([pos for pos in x], train_accuracies_sorted, width, label='Train Accuracy')
plt.bar([pos + width for pos in x], test_accuracies_sorted, width, label='Test Accuracy')
plt.bar([pos + 2*width for pos in x], test_recalls_sorted, width, label='Test Recall')

plt.xlabel("Models with Preprocessing and Scaling")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracies for Models (Sorted by Test Recall)")
plt.xticks(ticks=x, labels=x_labels_sorted, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# %%
data = list(zip(x_labels, test_recalls, test_precisions, test_f1s))
data_sorted = sorted(data, key=lambda x: x[3], reverse=True)

x_labels_sorted, test_precisions_sorted, test_recalls_sorted, test_f1_scores_sorted = zip(*data_sorted)

x = range(len(x_labels_sorted)) 

plt.figure(figsize=(40, 15))
plt.plot(x, test_precisions_sorted, marker='o', label='Test Precision', linestyle='-', linewidth=2)
plt.plot(x, test_recalls_sorted, marker='o', label='Test Recall', linestyle='-', linewidth=2)
plt.plot(x, test_f1_scores_sorted, marker='o', label='Test F1 Score', linestyle='-', linewidth=2)

plt.xlabel("Models with Preprocessing and Scaling")
plt.ylabel("Metric Value")
plt.title("Test Metrics (Precision, Recall, F1) for Models")
plt.xticks(ticks=x, labels=x_labels_sorted, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# %%
models = []
strategies = []
scalers = []
metrics = []
values = []

for model, model_data in results.items():
    for strategy, strategy_data in model_data.items():
        for scaler, scaler_data in strategy_data.items():
            for metric, value in scaler_data['Test'].items():
                if metric != 'Confusion Matrix':  # Skip confusion matrix
                    models.append(model)
                    strategies.append(strategy)
                    scalers.append(scaler)
                    metrics.append(metric)
                    values.append(value)

prepared_data = pd.DataFrame({
    'Model': models,
    'Strategy': strategies,
    'Scaler': scalers,
    'Metric': metrics,
    'Value': values
})
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC']
    
fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))

for i, metric in enumerate(metrics):
    metric_data = prepared_data[prepared_data['Metric'] == metric]
    
    sb.barplot(
        data=metric_data,
        x='Strategy',
        y='Value',
        hue='Model',
        ax=axes[i]
    )
    
    axes[i].set_title(f'{metric} by Model and Strategy')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylim(0, 1)
    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
plt.tight_layout()

# %%
models = prepared_data['Model'].unique()
fig, axes = plt.subplots(len(models), 1, figsize=(12, 8*len(models)))
if len(models) == 1:
    axes = [axes]

for i, model in enumerate(models):
    model_data = prepared_data[prepared_data['Model'] == model]
    
    pivot_df = model_data.pivot_table(
        values='Value',
        index=['Strategy', 'Scaler'],
        columns='Metric'
    )
    
    sb.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=axes[i]
    )
    axes[i].set_title(f'{model} Performance Metrics')

plt.tight_layout()

# %% [markdown]
# ## 5.3 Feature Importance & Double Checking For Fine-Tuning
# Analyze and explain the most important features:

# %%
results["LightGBM"]["SMOTE Data"]["No Scaling"]["Test"]

# %%
LGBM["SMOTE Data_No Scaling"].feature_importances_

# %%
lgb.plot_importance(LGBM["SMOTE Data_No Scaling"], importance_type="auto", figsize=(7,6), title="LightGBM Feature Importance")

# %%
RF["Under Sampled Data_Standard Scaler"].feature_importances_

# %% [markdown]
# ## 5.4 Final Model
# Choose the best-performing model based on your evaluations and fine-tuning.

# %%

param_grid = {
    'n_estimators': [150, 200, 250],  
    'criterion': ['entropy', 'gini'],  
    'max_depth': [25, 30, None],     
    'min_samples_split': [2, 3],     
    'min_samples_leaf': [1, 2],       
    'class_weight': ['balanced'],     
    'max_features': ['sqrt', 'log2'], 
    'bootstrap': [True, False],
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),# average='weighted'),
    'recall': make_scorer(recall_score),# average='weighted'),
    'f1': make_scorer(f1_score),# average='weighted'),
    'roc_auc': make_scorer(roc_auc_score)# average='weighted', multi_class='ovr', needs_proba=True)
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    min_weight_fraction_leaf=0.0,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
    oob_score=True 
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1',            # Refit on F1 since it balances precision and recall
    verbose=2,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(all_types_data["No Imbalance Strategy"]["Standard Scaler"]["X_train"]
                ,all_types_data["No Imbalance Strategy"]["Standard Scaler"]["y_train"])


n_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"Total parameter combinations to be tested: {n_combinations}")

# %%
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_accuracy"], cvres["params"]):
    print(mean_score, params)

# %%
results_ = pd.DataFrame(grid_search.cv_results_)
results_ = results_.sort_values(by='mean_test_f1', ascending=False)

top_10_models = results_.head(10)

heatmap_data = top_10_models[['mean_test_accuracy', 'mean_test_precision', 
                             'mean_test_recall', 'mean_test_f1', 'mean_test_roc_auc']]

plt.figure(figsize=(10, 6))
sb.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title('Top 10 Models by F1-Score')
plt.xlabel('Metrics')
plt.ylabel('Model Rank')
plt.show()

# %%
randomForest = grid_search.best_estimator_
grid_search.best_index_,grid_search.best_params_

# %%
param_grid_flgb = {
    'n_estimators': [300, 400, 500],
    'max_depth': [15, 20],
    'min_child_samples': [30],
    'num_leaves': [63, 127],
    'learning_rate': [0.05, 0.01],
    'colsample_bytree': [0.8, 0.9],
    'class_weight': ['balanced']
}

scoring_flgb = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),# average='weighted'),
    'recall': make_scorer(recall_score),# average='weighted'),
    'f1': make_scorer(f1_score),# average='weighted'),
    'roc_auc': make_scorer(roc_auc_score),# average='weighted', needs_proba=True)
}

lgbm = lgb.LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    boosting_type='gbdt'
)

grid_search_flgb = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid_flgb,
    cv=5,
    scoring=scoring_flgb,
    refit='f1',
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)
grid_search_flgb.fit(all_types_data["SMOTE Data"]["No Scaling"]["X_train"],
                     all_types_data["SMOTE Data"]["No Scaling"]["y_train"])

# %%
results_ = pd.DataFrame(grid_search_flgb.cv_results_)
results_ = results_.sort_values(by='mean_test_f1', ascending=False)

top_10_models = results_.head(10)

heatmap_data = top_10_models[['mean_test_accuracy', 'mean_test_precision', 
                             'mean_test_recall', 'mean_test_f1', 'mean_test_roc_auc']]

plt.figure(figsize=(10, 6))
sb.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title('Top 10 Models by F1-Score')
plt.xlabel('Metrics')
plt.ylabel('Model Rank')
plt.show()

# %%
grid_search_flgb.best_score_
grid_search.best_index_,grid_search.best_params_

# %%

lightGbm = grid_search_flgb.best_estimator_
y_pred = lightGbm.predict(all_types_data["SMOTE Data"]["No Scaling"]["X_test"])  
y_true = all_types_data["SMOTE Data"]["No Scaling"]["y_test"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of lightGbm')
plt.show()
randomForest = grid_search.best_estimator_
y_pred = randomForest.predict(all_types_data["SMOTE Data"]["Standard Scaler"]["X_test"])
y_true =all_types_data["SMOTE Data"]["Standard Scaler"]["y_test"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of Random Forrest')
plt.show()

# %%
y_pred_lgbm = lightGbm.predict(all_types_data["SMOTE Data"]["No Scaling"]["X_test"])  
y_pred_rf = randomForest.predict(all_types_data["SMOTE Data"]["Standard Scaler"]["X_test"])
y_true = all_types_data["SMOTE Data"]["No Scaling"]["y_test"]


# %%
fig = plot_model_comparison(y_true, y_pred_lgbm, y_pred_rf, 
                          model1_name="LightGBM", model2_name="Random Forrest")
plt.show()


# %%
y_pred_rf = randomForest.predict(all_types_data["SMOTE Data"]["Standard Scaler"]["X_test"])
y_pred_o_rf = RF["No Imbalance Strategy_Standard Scaler"].predict(all_types_data["SMOTE Data"]["Standard Scaler"]["X_test"])
y_true = all_types_data["SMOTE Data"]["Standard Scaler"]["y_test"]
fig = plot_model_comparison(y_true, y_pred_o_rf, y_pred_rf, 
                          model1_name="Random Forrest_old", model2_name="Random Forrest_final")


# %% [markdown]
# ## 5.6 Pickle the Model
# Save the final model for future deployment:

# %%
import pickle
with open('random_forrest_final.pkl', 'wb') as f:
    pickle.dump(randomForest, f)

with open('standart_scaler_final.plk','wb') as f:
    pickle.dump(standard,f)

# %% [markdown]
# ## <p style="background-color:#fea162; font-family:newtimeroman; color:#FFF9ED; font-size:175%; text-align:center; border-radius:10px 10px;">6. Conclusion</p>
# 
# <a id="6"></a>
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:blue; background-color:#dfa8e4" data-toggle="popover">Content</a>
# 
# ### Project Conclusion
# 
# In this project, we meticulously processed an unbalanced loan default dataset by implementing a comprehensive **data cleaning process**. This included the removal of unnecessary features and effective handling of missing values. Following a detailed **correlation analysis**, several features were merged to optimize the dataset's structure.
# 
# To address the data imbalance, we employed techniques such as **SMOTE ADASYN undersampling**. We also created variants of the dataset using **standard scaling, min-max scaling, and no scaling** to explore the impacts of different preprocessing methods.
# 
# A suite of nine machine learning models was applied:
# - KNN
# - Logistic Regression
# - Neural Network
# - Decision Tree
# - **Random Forest**
# - XGBoost
# - **LightGBM**
# - CatBoost
# 
# Through rigorous performance evaluations, **LightGBM** and **Random Forest** emerged as the top performers. We further fine-tuned these models through a grid search with carefully selected parameters, ultimately selecting **Random Forest** as the most suitable final model due to its superior performance. This model was saved, encapsulating a robust approach to predicting loan defaults. This approach leverages advanced ensemble techniques and adaptive preprocessing methods to ensure accuracy and reliability in predictions.

# %% [markdown]
# ......


