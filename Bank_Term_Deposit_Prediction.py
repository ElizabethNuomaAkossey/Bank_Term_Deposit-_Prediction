#!/usr/bin/env python
# coding: utf-8

# #  Bank Term Deposit Prediction With Machine Learning

# ## - Table of Contents
# 
# 1. [Data Description](#1.-Data-Description)
# 
# 2. [Importation of Packages](#2.-Importation-of-Packages)
# 
# 3. [Data Importation](#3.-Data-Importation)
# 
# 4. [Exploratory Data Analysis](#4.-Exploratory-Data-Analysis-(EDA))
#     
#     4.1 [Data Cleaning and Validation](#4.1-Data-Cleaning-and-Validation)
# 
#    4.2 [Univariate Analysis](#4.2-Univariate-Analysis)
#    
#    4.3 [Bivariate Analysis](#4.3-Bivariate-Analysis)
#    
#    4.4 [Multivariate Analysis](#4.4-Multivariate-Analysis)
#    
#    
# 5. [Feature Engineering and Data Preprocessing](#5.--Feature-Engineering-and-Data-Preprocessing)
# 
# 6. [Model Building](#6.--Model-Building)
# 
# 7. [Model Evaluation](#7.-Model-Evaluation)
# 
# 8. [Model Selection](#8.-Model-Selection)
# 
# 9. [Feature Importance](#9.-Feature-Importance)
# 
# 10. [Summary](#10.-Summary)
# 
# 11. [Recommendations](#11.-Recommendations)
# 
# 12. [Model Deployment](#12.--Model-Deployment)

# ### 1. Data Description
# The data is related to direct marketing campaigns (phone calls) of a banking institution.  
# The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# Data Source: https://fromsmash.com/ZLq5W.CQkL-ct
# 
# 
#    #### Bank Client Data:
#    
#    1 - **Age** (numeric)
#    
#    2 - **Job:** type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services") 
#                                        
#    3 - **Marital:** marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
#    
#    4 - **Education** (categorical: "unknown","secondary","primary","tertiary")
#    
#    5 - **Default:** has credit in default? (binary: "yes","no")
#    
#    6 - **Balance:** average yearly balance, in euros (numeric) 
#    
#    7 - **Housing:** has housing loan? (binary: "yes","no")
#    
#    8 - **Loan:** has personal loan? (binary: "yes","no")
#    
#    #### Related with the last contact of the current campaign:
#    
#    9 - **Contact:** contact communication type (categorical: "unknown","telephone","cellular") 
#    
#   10 - **Day:** last contact day of the month (numeric)
#   
#   11 - **Month:** last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   
#   12 - **Duration:** last contact duration, in seconds (numeric)
#   
#    #### Other attributes:
#    
#   13 - **Campaign:** number of contacts performed during this campaign and for this client (numeric, includes last contact)
#   
#   14 - **Pdays:** number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#   
#   15 - **Previous:** number of contacts performed before this campaign and for this client (numeric)
#   
#   16 - **Poutcome:** outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
#   #### Output variable (desired target):
#   
#   17 - **y:** has the client subscribed a term deposit? (binary: "yes","no")
# 

# ### 2. Importation of Packages
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[1]:


#importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing package data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

#importing package to balance data
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

# Importing Classifcation Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Importing package for metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


# In[2]:


pip install xgboost


# ### 3. Data Importation
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[3]:


df = pd.read_csv("bank-full.csv")


# ### 4. Exploratory Data Analysis (EDA)
# 
# [Back to Table of Contents](#--Table-of-Contents)

# #### 4.1 Data Cleaning and Validation

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# #### Splitting Column 

# In[10]:


#splitting column
column_names= 'age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'

df[["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact","day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]] = df[column_names].str.replace('"','').str.split(";", expand = True)

#drop the original column
df.drop(columns =column_names, inplace = True)


# #### Sanity Checks after Splitting Column

# In[11]:


df.head(30)


# In[12]:


df.tail()


# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df.describe()


# In[16]:


#Checking for null values
df.isnull().sum()


# In[17]:


#Checking for duplicate values
duplicates = df.duplicated()
duplicates.sum()


# #### Changing Datatypes of Some Columns

# In[18]:


df = df.astype({
                'age':'int',
                'balance':'float',
                'day':'int',
                'duration':'int',
                'campaign':'int',
                'pdays':'int',
                'previous':'int'
                })


# In[19]:


df.info()


# In[20]:


df["age"].nunique()


# In[21]:


df["job"].value_counts()


# In[22]:


df["marital"].unique()


# In[23]:


df["default"].unique()


# In[24]:


df["balance"].nunique()


# In[25]:


df["housing"].unique()


# In[26]:


df["loan"].unique()


# In[27]:


df["contact"].unique()


# In[28]:


df["day"].nunique()


# In[29]:


df["month"].value_counts()


# In[30]:


df["duration"].nunique()


# In[31]:


df["campaign"].unique()


# In[32]:


df["pdays"].nunique()


# In[33]:


df["poutcome"].unique()


# In[34]:


df["y"].unique()


# In[35]:


#describing the categorical variables
df.describe(include='object')


# In[36]:


df.describe(exclude='object')


# In[37]:


#Checking for outliers in the numerical variables

fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(15,30))
plt.title('outliers in numerical data')

sns.boxplot(df['age'],color='skyblue',ax=axes[0,0])
axes[0,0].set_title('age')

sns.boxplot(df['balance'],color='salmon',ax=axes[0,1])
axes[0,1].set_title('balance')

sns.boxplot(df['day'],color='indigo',ax=axes[0,2])
axes[0,2].set_title('day')

sns.boxplot(df['duration'],color='green',ax=axes[1,0])
axes[1,0].set_title('duration')

sns.boxplot(df['campaign'],color='violet',ax=axes[1,1])
axes[1,1].set_title('campaign')

sns.boxplot(df['pdays'],color='orange',ax=axes[1,2])
axes[1,2].set_title('pdays')

sns.boxplot(df["previous"],color='green', ax=axes[2,0])
axes[2,0].set_title("previous")

axes[2,1].axis('off')
axes[2,2].axis('off')

plt.tight_layout()
plt.show()


# In[38]:


#Finding the correlation between numerical variables

df.select_dtypes("number").corr()


# In[39]:


# Create correlation matrix
correlation = df.select_dtypes("number").corr()
correlation

# Plot heatmap of `correlation`
sns.heatmap(correlation,annot = correlation,cmap = 'Spectral_r');


# #### 4.2 Univariate Analysis 
# 
# Numerical
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[40]:


fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(15,30))

sns.histplot(df['age'],kde=True, color='skyblue',ax=axes[0,0])
axes[0,0].set_title('age')

sns.histplot(df['balance'],kde=True, color='salmon',ax=axes[0,1])
axes[0,1].set_title('balance')

sns.histplot(df['day'],kde=True,color='indigo',ax=axes[0,2])
axes[0,2].set_title('day')

sns.histplot(df['duration'],kde=True,color='green',ax=axes[1,0])
axes[1,0].set_title('duration')

sns.histplot(df['campaign'],kde=True,color='violet',ax=axes[1,1])
axes[1,1].set_title('campaign')

sns.histplot(df['pdays'],kde=True,color='orange',ax=axes[1,2])
axes[1,2].set_title('pdays')

sns.histplot(df["previous"],kde=True,color='green', ax=axes[2,0])
axes[2,0].set_title("previous")

axes[2,1].axis('off')
axes[2,2].axis('off')

plt.tight_layout()
plt.show()


# 
# #### Categorical
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[41]:


fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(25,30))

sns.countplot(x="job" , data =df, palette='Set1',ax=axes[0,0])
axes[0,0].set_title('job')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)


sns.countplot(x="marital",data=df,palette='Set1',ax=axes[0,1])
axes[0,1].set_title('marital')

sns.countplot(x="education",data=df,palette='Set1',ax=axes[0,2])
axes[0,2].set_title('education')

sns.countplot(x="default",data=df,palette='Set1',ax=axes[1,0])
axes[1,0].set_title('default')

sns.countplot(x='housing',data=df,palette='Set1',ax=axes[1,1])
axes[1,1].set_title('housing')

sns.countplot(x="loan",data=df,palette='Set1',ax=axes[1,2])
axes[1,2].set_title('loan')

sns.countplot(x="contact",data=df,palette='Set1', ax=axes[2,0])
axes[2,0].set_title("contact")

sns.countplot(x="month",data=df,palette='Set1', ax=axes[2,1])
axes[2,1].set_title("month")


sns.countplot(x="poutcome",data=df,palette='Set1', ax=axes[2,2])
axes[2,2].set_title("poutcome")

sns.countplot(x="y",data=df,palette='Set1', ax=axes[3,0])
axes[3,0].set_title("y")


axes[3,1].axis('off')
axes[3,2].axis('off')

plt.tight_layout()
plt.show()


# #### 4.3 Bivariate Analysis 
# 
# [Back to Table of Contents](#--Table-of-Contents)
# 

# In[42]:


numerical_vars = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

for num_var in numerical_vars:
    for cat_var in categorical_vars:
        plt.figure(figsize=(10, 7))
        sns.barplot(x=cat_var, y=num_var, data=df,estimator='mean',errorbar =None)
        plt.title(f'{num_var} by {cat_var}')
        plt.xticks(rotation=45)
        plt.show()


# In[43]:


sns.histplot(df[df['y']=='yes']['age'])
plt.show()


# In[44]:


mean_balance_per_job = df.groupby(['job'])['balance'].mean().sort_values(ascending=False)

print(mean_balance_per_job)
sns.barplot(x=mean_balance_per_job.index, y=mean_balance_per_job.values)
plt.xticks(rotation=90)
plt.show()


# #### 4.4 Multivariate Analysis
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[45]:


sns.pairplot(df[['age', 'balance', 'duration', 'campaign','pdays', 'previous']])
plt.title('Pairplot of Numerical Variables')
plt.show()


# In[46]:


sns.barplot(x='housing', y='balance', hue='y', data=df)
plt.title('Average Balance by Loan Status and Housing')
plt.show()


# In[47]:


sns.barplot(x='loan', y='age', hue='y', data=df)
plt.title('Average Age by Loan Status and Taget Variable(y)')
plt.show()


# In[48]:


sns.barplot(x='loan', y='balance', hue='y', data=df)
plt.title('Average Balance by Loan Status and Housing')
plt.show()


# In[49]:


sns.barplot(x='job', y='age', hue='y', data=df)
plt.title('Average Age by Job Status and Subscription to a Term Deposit')
plt.xticks(rotation=45)
plt.show()


# In[50]:


sns.barplot(x='poutcome', y='previous', hue='y', data=df)
plt.title('Average Number of contacts performed before the campaign and foe the Client by Previous outome Status and Subscription to a Term Deposit')
plt.xticks(rotation=45)
plt.show()


# In[51]:


sns.barplot(x='poutcome', y='duration', hue='y', data=df)
plt.title('Average Duration by Previous Outcome and Subscription to a Term Deposit')
plt.xticks(rotation=45)
plt.show()


# In[52]:


sns.barplot(x='job', y='duration', hue='y', data=df)
plt.title('Average Duration by Job and Subscription to a Term Deposit')
plt.xticks(rotation=45)
plt.show()


# In[53]:


sns.scatterplot(data=df,x='age',y='balance',hue='y')
plt.title("Age vs Balance with Subscription Outcome")
plt.show()


# In[54]:


sns.barplot(x='job', y='age', hue='y', data=df,errorbar=None)
plt.title('Average Age by Job and Subscription to a Term Deposit')
plt.xticks(rotation=45)
plt.show()


# In[55]:


#Calculating the percentage of each value of the target variable
df['y'].value_counts(normalize=True)*100


# In[56]:


Count_of_Balance_lessthan0_per_Job = df[df['balance']<0]['job'].value_counts()
sns.barplot(x=Count_of_Balance_lessthan0_per_Job.index, y=Count_of_Balance_lessthan0_per_Job.values)
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.xlabel('Jobs')
plt.title('Frequency of Jobs with Balance less than 0')
plt.show()


# In[57]:


Count_of_Balance_greaterthan0_per_Job = df[df['balance']>1000]['job'].value_counts()
sns.barplot(x=Count_of_Balance_greaterthan0_per_Job.index, y=Count_of_Balance_greaterthan0_per_Job.values)
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.xlabel('Jobs')
plt.title('Frequency of Jobs with Balance Greater than 0')
plt.show()


# In[58]:


df.groupby(['default','housing','loan'])['balance'].mean()


# In[59]:


df[df['y'] == 1].groupby('month').size().sort_values(ascending=False)


# In[60]:


df.groupby(['y','housing','loan','default'])['balance'].mean()


# In[61]:


df.groupby(['y','housing'])['duration'].mean()


# In[62]:


df.groupby(['y','loan'])['balance'].mean()


# In[63]:


# Average balance by marital group
df.groupby(['marital'])['balance'].mean()


# In[64]:


#Average balance by marital group and job
df.groupby(['marital','job'])['campaign'].mean()


# In[65]:


#Dealing with ouliers in the balance variable
Q1 = df.balance.quantile(0.25)
Q3 = df.balance.quantile(0.75)

IQR = Q3 - Q1

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR
print(lower,upper,sep=' ,')


# In[66]:


#Removing balance outliers
not_outlier_df= df[(df['balance']>lower)&(df['balance']<upper)]
not_outlier_df


# In[67]:


#Plotting a box plot after removing outliers in the balance column
sns.boxplot(not_outlier_df['balance'])
plt.show()


# In[68]:


#Removing Redundant features in the dataset
df.drop(columns=['default','contact','pdays','poutcome'],inplace=True)
df


# ### 5.  Feature Engineering and Data Preprocessing
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[69]:


# Counting the number of each instances in the target variable
sns.countplot(x="y", data=df, palette="Set1")
plt.title("Count of Y")
plt.show()


# In[70]:


# renaming target variable
df["y"] = df["y"].replace({"yes": 1, "no": 0})


# In[71]:


df["y"].value_counts()


# In[72]:


df.describe(include='object')


# In[73]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
encoder = LabelEncoder()

# List of columns to encode
columns_to_encode = ['job', 'marital', 'education', 'housing', 'month','loan']

# Apply LabelEncoder to each column in the list
for column in columns_to_encode:
    df[column] = encoder.fit_transform(df[column])


# In[74]:


df['loan'] = encoder.fit_transform(df['loan'])


# In[75]:


df.head()


# In[76]:


# Splitting the data into features and target variable
X = df.drop(columns=['y'], axis=1) 
y = df['y'] 

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[77]:


# Visualizing original target variable distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train)
plt.title('Original Class Distribution (Before Balancing)')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()


# In[78]:


#Counting the number of each target variable
y_train.value_counts()


# In[79]:


#Balancing the target variable
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[80]:


#Visualizing the distribution of the target variable after balancing the data
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTE (Balancing)')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()


# In[81]:


y_train_resampled.value_counts()


# ### 6.  Model Building
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[82]:


# Function to evaluate models
def evaluate_model(model, X_train_resampled, X_test, y_train_resampled, y_test):
    model.fit(X_train_resampled, y_train_resampled)
    y_train_pred = model.predict(X_train_resampled)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train_resampled, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Train F1 Score: {train_f1:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    print('Classification Report (Test Data):')
    print(classification_report(y_test, y_test_pred))
    
    print('Confusion Matrix (Test Data):')
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
    plt.show()
    
    return train_accuracy, test_accuracy, train_f1, test_f1


# In[83]:


# Calculate scale_pos_weight as the ratio of negative to positive class
scale_pos_weight = sum(y_train_resampled == 0) / sum(y_train_resampled == 1)
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'NaiveBayes': GaussianNB(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

param_grids = {
    'DecisionTree': {'max_depth': [3, 5, 7, 9], 'class_weight': ['balanced', None]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9],'class_weight': ['balanced', None]},
    'ExtraTrees': {'n_estimators':[10,20,30,50,100],'class_weight': ['balanced', None]},
    'NaiveBayes':{},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2],'class_weight': ['balanced', None]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

results = []


# ### 7. Model Evaluation
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[84]:


# Using Grid search to find the best hyperparameter for each model
for model_name, model in models.items():
    print(f'\nModel: {model_name}')
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')
    
    train_acc, test_acc, train_f1, test_f1 = evaluate_model(best_model, X_train_resampled, X_test, y_train_resampled, y_test)
    
    results.append({
        'Model': model_name,
        'Best Params': grid_search.best_params_,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train F1 Score': train_f1,
        'Test F1 Score': test_f1
    })


# ### 8. Model Selection
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[85]:


results_df = pd.DataFrame(results)
print(results_df)

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

metrics = ['Train Accuracy', 'Test Accuracy', 'Train F1 Score', 'Test F1 Score']

for idx, metric in enumerate(metrics):
    sns.barplot(x='Model', y=metric, data=results_df, ax=axes[idx], palette='viridis')
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].set_ylabel(metric)
    axes[idx].set_xlabel('Model')
    for item in axes[idx].get_xticklabels():
        item.set_rotation(45)

plt.tight_layout()
plt.show()


# In[86]:


results_df = pd.DataFrame(results)
print(results_df)
# Finding the best model based on Test Accuracy and Test F1 Score
best_model_accuracy_row = results_df.loc[results_df['Test Accuracy'].idxmax()]
best_model_f1_row = results_df.loc[results_df['Test F1 Score'].idxmax()]

print("Best Model based on Test Accuracy:")
print(best_model_accuracy_row)

print("\nBest Model based on Test F1 Score:")
print(best_model_f1_row)

results_df['Average Score'] = (results_df['Test Accuracy'] + results_df['Test F1 Score']) / 2
best_model_combined_row = results_df.loc[results_df['Average Score'].idxmax()]

print("\nBest Model based on Combined Test Accuracy and Test F1 Score:")
print(best_model_combined_row)


# #Counting the number if correct and incorrect prediction of the selected model
# best_xgb_params = {'learning_rate': 0.1, 'n_estimators': 200}
# 
# xgb_model = XGBClassifier(**best_xgb_params)
# 
# xgb_model.fit(X_train_resampled, y_train_resampled)
# 
# y_test_pred = xgb_model.predict(X_test)
# 
# # DataFrame to show true vs predicted values
# prediction_df = pd.DataFrame({
#     'True Label': y_test,
#     'Predicted Label': y_test_pred
# })
# 
# prediction_df['Correct Prediction'] = prediction_df['True Label'] == prediction_df['Predicted Label']
# 
# print(prediction_df.head(50))
# 
# prediction_df.value_counts()

# In[88]:


#ROC Curve

y_test_probs = xgb_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)
roc_auc = roc_auc_score(y_test, y_test_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ### 9. Feature Importance
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[89]:


#Getting feature importance from the model selected
feature_importances = xgb_model.feature_importances_

# Creating a DataFrame 
feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sorting the features by order of importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.show()


# ### 10. Summary
# 
# [Back to Table of Contents](#--Table-of-Contents)
# 
# The data 1 column and 45211 rows. It was splitted into 17 columns by 45211 rows with no missing values or duplicates after the sanity checks. Exploratory analysis was carried out and the results described below;
#  
# * From the exploratory analysis, Age was found to be an important feature of clients who will subscribe to a term deposit, with middle-aged clients of about 30-35 years, like subscribe.
# 
# 
# * Also, Customers in jobs like "blue-collar", "technician", and "retired" showed a higher likelihood of subscribing to a term deposit although quite a number of them has less balance in their account.
# 
# * Again, Customers with a higher account Balance were more likely to subscribe to a term deposit.
# 
# * Customers with a higher Education level, especially those with tertiary education, had a higher likelihood of subscribing.
# 
# * Previous contacted customers during the campaign whad a higher chance of subscribing.
# 
# * Customers who had no loan or housing loan also were more likely to subscribe to a term deposit.
# 
# * In the month May and June, the subscription rates were high which indicates the season in which most customers were likely to subscribe to a term deposit.
# 
# * Singles and divorced Customers also showed interest in subscribing to a term deposit more than married Customers, probably they have a lot of responsibilities to cater for.
# 
# * The higher the duration,that is; the longer the time spent on the call, the higher, Customers subscribed to a term deposit.
# 
# 2. It was also observed that the target variable, was imbalanced as about 88% of it were "No" whiles about 11% was "Yes". In order to avoid avoid bias in the model, SMOTE and class_weight was used to handle it.
# 
# Also, redundant features were dropped to avoid the model from underfitting or overfitting.
# 
# About 6 Algorithms were used in the model building, hyperparameter tuning was done and the best model selected based on their Accuracy and F1 score.
# 
# The best performing model selected was Xgboost with a test Accuracy of 87% and test F1-Score of 56%.
# 
# Though the data is overfiting, this is due to inadequate data and unbalanced presentation of the target variable during testing.

# ### 11. Recommendations
# 
# [Back to Table of Contents](#--Table-of-Contents)
# 
# 1. Marketting should be centered on customers of middle aged (30-50), especially those with higher educational qualifications. And adverisement should be based on the long term essence of subscribing to the term deposit.
# 
# 2. Those with blue collar jobs and technicians should be highly considered, for example; a message can be tailored or personalized to them to constantly remind and teach them essence of term deposit.
# 
# 3. Utilize seasonal campaigns during peak months for example in  May and June as seen from the analysis when subscription rates are higher.
# 
# 4. Interact more with clients who were previously contacted in the campaign, especially those who were interested but did not subscribe. Example, the company can offer them nice incentives and probably additional information that will enthuse them to subscibe the term deposit. 

# ### 12.  Model Deployment
# 
# [Back to Table of Contents](#--Table-of-Contents)

# In[91]:


import pickle


# In[92]:


# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)


# In[95]:


import pickle
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("The model file is missing!")
except Exception as e:
    print(f"Error loading model: {e}")


# [Back to Table of Contents](#--Table-of-Contents)

# In[ ]:




