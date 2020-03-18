import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score,\
precision_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns


"""Problem Statement
Client: Tax firm looking to sell tax preparation software

Data: Two years of customer information and record whether they were able to
sell successfully to each customer.

Goal: They want to understand this data and build a model to predict if they
will be able to successfully sell their software to a given individual.

Approach:
    1. Exploratory data analysis to identify features that contribute to a
successful sale (successful_sell == 'yes')
    2. Two classification models to predict whether a sale would be successful
or not
"""

"""Exploratory Data Analysis
Uninformative features and duplicate records will be deleted.
Checked class balance/imbalance in target column 'successful_sell'
Analysis will focus on the following identifable features:
    - Age
    - Day of the week (dow)
    - Employment status
    - Marriage status
    - Month
    - School level
Any interesting unidentifiable features that arise during modeling will be
revisited and explored.
"""

#Loading and reading data
df = pd.read_csv("project_data.csv")
print(df.info())
# There were 41188 customer records over the last two years.

''' Uninformative feature found and deleted:
There appears to be a low number of values for c8, so it's probably not
informative to the successful_sell column. It has been deleted from the dataset.
'''

no_c8_val = df['c8'].isnull().sum()
no_c8 = df['c8'].isnull().sum()/len(df['c8'])
print('Number of null c8 values: {} \nPercentage of Null c8 Values: {}%'.format(no_c8_val, 100*np.round(no_c8, 3)))
df = df.drop('c8', axis = 1)

"""Checking for duplicates"""
print("Length of duplicated records: ",len(df[df.duplicated()]))
#There are no duplicate rows. So no records were deleted.

"""Checking for target class imbalance in 'successful_sale' column """
print(pd.concat([df['successful_sell'].value_counts(), df['successful_sell'].value_counts(normalize=True)], keys = ['Counts', 'Record_%'], axis = 1),'\n\n')
#The classes are highly imbalanced.

"""Aggregations Performed on Successful Sales
Approach:
1. Current customer target: total in the record
2. Ideal customer market: only those who have succesful sales

Implementation:
1. Map successful sell columns to floats
2. Pandas.groupby mean() --> Ratio of successful sales
3. Pandas.groupby sum() --> Total number of successful sales"""

#Mapped successful sell columns to floats
df['successful_sell']=df['successful_sell'].map({'yes':1, 'no':0})


"""Employment - Successful Sale Relationship"""

print("Employment - Successful Sale Relationship")
print("Current Customer Market - Employment")
print(pd.concat([df['employment'].value_counts(), df['employment'].value_counts(normalize=True)], keys = ['Counts', 'Normalized_counts'], axis = 1),'\n\n')

print("Ideal Customer Market - Employment")
#Employment aggregation
emp_group =df.groupby(by = 'employment', axis=0, as_index = False)
emp_group.successful_sell.mean().sort_values('successful_sell', ascending = False).reset_index(drop = True)

#Number of successful sales per employment value
emp_count = emp_group.successful_sell.sum()

#Ratio of successful sales per employment value
emp_mean = emp_group.successful_sell.mean()

emp_df = pd.concat([emp_count, 100*np.round(emp_mean.iloc[:,1], 3)], axis = 1)
emp_df.columns = ['Employment', 'Count', 'Percentage']

#Sorted by ratio of successful sales per employment value
emp_df=emp_df.sort_values(by='Percentage', axis = 0, ascending = False)
print(emp_df)

"""Marriage Status - Successful Sale Relationship"""

print("\n\nMarriage - Successful Sale Relationship")
print("Current Customer Market - Marriage Status")
print(pd.concat([df['marriage-status'].value_counts(), df['marriage-status'].value_counts(normalize=True)], keys = ['Counts', 'Normalized_counts'], axis = 1),'\n\n')

#Marriage group aggregation
marriage_group =df.groupby(by = 'marriage-status', axis=0, as_index = False)
print("Ideal Customer Market - Marriage Status")
#Number of successful sales per marriage status value
marriage_count = marriage_group.successful_sell.sum()

#Ratio of successful sales per marriage status value
marriage_mean = marriage_group.successful_sell.mean()

#Sorted by ratio of successful sales per marriage status value
print(pd.concat([marriage_count, 100*np.round(marriage_mean.iloc[:,1],3)], axis = 1))




"""DOW - Successful Sale Relationship"""

print("\n\nDOW - Successful Sale Relationship")
print("Current Customer Market - DOW")
print(pd.concat([df['dow'].value_counts(), df['dow'].value_counts(normalize=True)], keys = ['Counts', 'Normalized_counts'], axis = 1),'\n\n')

#DOW group aggregation
dow_group =df.groupby(by = 'dow', axis=0, as_index = False)
print("Ideal Customer Market - DOW")
#Number of successful sales per DOW status value
dow_count = dow_group.successful_sell.sum()

#Ratio of successful sales per DOW status value
dow_mean = dow_group.successful_sell.mean()

#Sorted by ratio of successful sales per DOW status value
print(pd.concat([dow_count, dow_mean.iloc[:,1]], axis = 1))

#There appears to be no appreciable difference of sales performance across dow values


print("\n\nSchool - Successful Sale Relationship")
print("Current Customer Market - School")
print(pd.concat([df['school'].value_counts(), df['school'].value_counts(normalize=True)], keys = ['Counts', 'Normalized_counts'], axis = 1),'\n\n')

print("Ideal Customer Market - School")
#Employment aggregation
school_group =df.groupby(by = 'school', axis=0, as_index = False)
school_group.successful_sell.mean().sort_values('successful_sell', ascending = False).reset_index(drop = True)

#Number of successful sales per employment value
school_count = school_group.successful_sell.sum()

#Ratio of successful sales per employment value
school_mean = school_group.successful_sell.mean()

school_df = pd.concat([school_count, 100*np.round(school_mean.iloc[:,1], 3)], axis = 1)
school_df.columns = ['Employment', 'Count', 'Percentage']

#Sorted by ratio of successful sales per employment value
school_df=school_df.sort_values(by='Percentage', axis = 0, ascending = False)
print(school_df)


n4_group =df.groupby(by = 'n4', axis=0, as_index = False)
print('\n\nPercentage of n4 Data Points')
print(df['n4'].value_counts(normalize=True),'\n\n')
#More than 96% of n4 values are the same. Since n4 appears to be an uninformative feature, it will be dropped from further analysis.
df = df.drop('n4', axis = 1)


"""Month - Successful Sale Relationship"""

print("\n\nMonth - Successful Sale Relationship")
print("Current Customer Market - Month")
print(pd.concat([df['month'].value_counts(), df['month'].value_counts(normalize=True)], keys = ['Counts', 'Normalized_counts'], axis = 1),'\n\n')

print("Ideal Customer Market - Month")
#month aggregation
month_group =df.groupby(by = 'month', axis=0, as_index = False)
month_group.successful_sell.mean().sort_values('successful_sell', ascending = False).reset_index(drop = True)

#Number of successful sales per month value
month_count = month_group.successful_sell.sum()

#Ratio of successful sales per month value
month_mean = month_group.successful_sell.mean()

month_df = pd.concat([month_count, 100*np.round(month_mean.iloc[:,1], 3)], axis = 1)
month_df.columns = ['Month', 'Count', 'Percentage']

#Sorted by ratio of successful sales per month value
month_df=month_df.sort_values(by='Percentage', axis = 0, ascending = False)
print(month_df)


"""Age
Does a successful sale depend on age of the client?
To make age usable for the describe method, the age was changed from int to float.
"""
df['age'] = df['age'].astype(float)
print('\nAge Statistics of Current Market')
print(df['age'].describe())
print('\nAge Statistics of Ideal Market')
print(df[df['successful_sell']==1]['age'].describe())

#Age distribution plots
age_yes=df['age'][df['successful_sell']==1]
age_no=df['age'][df['successful_sell']==0]

sns.distplot(age_no, label = 'No', bins = 20)
sns.distplot(age_yes, label = 'Yes', bins = 20)

plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Impact of Age on Successful Sells')
plt.legend(prop = {'size':12})
plt.savefig("Age_Analysis.png")
plt.show()



'''n5
Does the presence of n5 change whether your customers buy or not?
The n5 factor doesn't appear to be different whether people buy or not.
'''
#n5
n5_yes=df['n5'][df['successful_sell']==1]
n5_no=df['n5'][df['successful_sell']==0]

sns.distplot(n5_no, label = 'No')
sns.distplot(n5_yes, label = 'Yes')

plt.title('Impact of n5 on Successful Sales')
plt.xlabel('n5')
plt.ylabel('Density')
plt.legend(prop = {'size':12})
plt.savefig('n5_populations.png')
plt.show()



"""Sting to Numeric Value Conversion.
Reason: scikit-learn models require numerical value inputs"""

# Gathering unique values to gauge feature engineering stategies for the strings.
for each in df.columns:
    print(each, df[each].unique())


""" Converting String Values to Numerical Features

Features dataset (df_features) mapping
Conversion to imply numerical order

- b1: yes = 1, no = 0
- b2: yes = 1, no = 0, NaN = -1
- c10: yes = 1, no = 0
- c3: True = 1, False = 0, unknown = -1
- c4: new = 1, old = 0
- dow (day of the week): mon = 0, tues = 1, wed = 2, thurs = 3, fri = 4
- month: jan = 0, feb = 1, ... nov = 10, dec = 12
- school: 5 - a decent amount = 5, 5 - a lot = 5, 2 - a little bit = 2, 4 - average amount = 4, 3 - a bit more = 3, 1 - almost none = 1, 0 - none = 0, NaN = -1

"""


df_features = df.copy().drop('successful_sell', axis = 1)
df_features['b1'] = df['b1'].map({'yes':1, 'no':0, '-1':-1})
df_features['b2'] = df['b2'].fillna(-1).map({'yes':1, 'no':0, -1:-1})
df_features['c10'] = df['c10'].map({'yes':1, 'no':0})
df_features['c3'] = df['c3'].map({'True':1, 'False':0, 'unknown':-1})
df_features['c4'] = df['c4'].map({'new':1, 'old':0})
df_features['dow'] = df['dow'].map({'mon':0, 'tues':1, 'wed':2, 'thurs':3, 'fri':4}).fillna(-1)
#Month map
months = {'jan':0, 'feb':1, 'mar':2, 'apr':3, 'may':4, 'jun':5, 'jul':6, 'aug':7, 'sep':8, 'oct':9, 'nov':10, 'dec':11}
df_features['month']=df_features['month'].map(months)
#School map
schools = {'5 - a decent amount': 5, '5 - a lot':5, '2 - a little bit':2, '4 - average amount': 4, '3 - a bit more':3, '1 - almost none':1, '0 - none':0}
df_features['school'] = df_features['school'].map(schools).fillna(-1)


'''String Features Conversion II - Without Order
 - Marriage-status
 - Employment

 Any NaN marriage-status values were converted to unknown. Both the marriage status and employment
 columns were one-hot encoded in order to generate numerical features without implying order. This
 expanded the feature space to 35.'''

#Marriage status and Employment will be one hot decoded, since there is no order
df_features['marriage-status'] = df_features['marriage-status'].fillna('unknown')
df_features = pd.get_dummies(df_features, columns = ['marriage-status', 'employment'])
#df_features.info()


"""Preparing Training, Testing, and Validation Datasets
1. Splitting featurespace into training/test/validation datasets
2. Scaling the data to zero mean and unit variance"""

X = df_features
y = df.successful_sell

#Two sequential splits to produce a 60/20/20 training/test/validation split.
"""Since the target classes are so imbalanced, the stratify flag is used in the train_test_split function
to replicate the same imbalance for the models."""
#First split: Splits 20% of the data into validation dataset
X_model, X_val, y_model, y_val = train_test_split(X, y, train_size = 0.8, stratify=y)
#Second split: Splits the remaining into training and test
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, train_size = 0.75, stratify=y_model)

print('The training dataset contains {} samples or {}% of the entire dataset.'.format(X_train.shape[0], 100*np.round(X_train.shape[0]/X.shape[0], 3)))
print('The test dataset contains {} samples or {}% of the entire dataset.'.format(X_test.shape[0], 100*np.round(X_test.shape[0]/X.shape[0], 3)))
print('The validation dataset contains {} samples or {}% of the entire dataset.'.format(X_val.shape[0], 100*np.round(X_val.shape[0]/X.shape[0], 3)))


#Scaling the feature dataset to zero mean and unit variance
scaler=StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


#Model 1: Random Forest Classifier
print('Model #1 (Random Forest Classifier) Results')
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print("Random Forest Classifier Results on Test Data")
print(classification_report(y_test, y_pred))
print("Recall: ",recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1: ", f1_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_pred))

print('\nRandom Forest Classifier on Training Data Results')
print(classification_report(y_train, y_pred_train))
print("Recall: ",recall_score(y_train, y_pred_train))
print("Precision: ", precision_score(y_train, y_pred_train))
print("Confusion Matrix: ", confusion_matrix(y_train, y_pred_train))
print("Accuracy: ", accuracy_score(y_train, y_pred_train))
print("F1: ", f1_score(y_train, y_pred_train))
print("AUC: ", roc_auc_score(y_train, y_pred_train))

print('\nRandom Forest Classifier on Validation Data Results')
y_pred_val = clf.predict(X_val)
print(classification_report(y_val, y_pred_val))
print("AUC: ", roc_auc_score(y_train, y_pred_train))


"""Feature Importances from Random Forest Classifier"""
# Create list of top most features based on importance
feature_names = X_model.columns
feature_imports = clf.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
indices = np.argsort(feature_imports)[::-1]

# Print the feature importance and feature name in descending order from highest to lowest importance values.
print("Feature ranking:")
for f in range(5):
    print("%d.  "% (f+1) + feature_names[indices[f]] +"   (%f)" % (feature_imports[indices[f]]))

#Plotting the feature importances
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Gini Importance')
plt.title('Most important features - Random Forest')
plt.savefig('RF_feature_import.png')
plt.show()

'''The most important feature was c10. So I dedicated this section to explore this feature.'''
# Importance of c10
c10_group =df.groupby(by = 'c10', axis=0, as_index = False)
c10_means = c10_group.successful_sell.mean()
c10_means.columns = ['c10', 'Sell %']

c10_counts = c10_group.successful_sell.count()
c10_counts.columns = ['c10', 'Counts']

print('Statistics of c10 Feature:')
print(pd.concat([c10_counts, c10_means.iloc[:,1]], axis = 1))



# # Model 2 : Logistic Regression
print('Model #2 (Logistic Regression) Results')
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)
y_pred_log = clf_log.predict(X_test)
y_pred_train_log = clf_log.predict(X_train)

print("Logistic Regression Results on Test Data")
print(classification_report(y_test, y_pred_log))
print("Recall: ",recall_score(y_test, y_pred_log))
print("Precision: ", precision_score(y_test, y_pred_log))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_log))
print("Accuracy: ", accuracy_score(y_test, y_pred_log))
print("F1: ", f1_score(y_test, y_pred_log))
print("AUC: ", roc_auc_score(y_test, y_pred_log),'\n\n\n')

print("Logistic Regression Results on Training Data")
print(classification_report(y_train, y_pred_train_log))
print("Recall: ",recall_score(y_train, y_pred_train))
print("Precision: ", precision_score(y_train, y_pred_train_log))
print("Confusion Matrix: ", confusion_matrix(y_train, y_pred_train_log))
print("Accuracy: ", accuracy_score(y_train, y_pred_train_log))
print("F1: ", f1_score(y_train, y_pred_train_log))
print("AUC: ", roc_auc_score(y_train, y_pred_train_log),'\n\n\n')

print("Logistic Regression Results on Validation Data")
y_pred_val_log = clf.predict(X_val)
print(classification_report(y_val, y_pred_val_log))
