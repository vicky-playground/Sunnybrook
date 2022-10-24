#!/usr/bin/env python
# coding: utf-8

# In[1071]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[1072]:


# load the datasets
df_data = pd.read_excel('activity data.xlsx')
df_900 = pd.read_excel('activity900.xlsx')


# In[1073]:


df_data.shape


# In[1074]:


df_data.isna().sum()


# In[1075]:


# remove the unncessary columns and missing values
df_data = df_data.drop(["StudyID","Performed","OtherDescription",'StudyVisit',"DateReceived","ActivityTypeCode","HealthProblemNote"],axis=1)
df_data = df_data[df_data['ComplexClassification'].notna()]
df_data = df_data[df_data['Months'].notna()]
df_data = df_data[df_data['AvgTimesMonth'].notna()]
df_data = df_data[df_data['HipPain'].notna()]
df_data = df_data[df_data['ReducedTime'].notna()]
df_data = df_data[df_data['HealthProblem'].notna()]
df_data = df_data[(df_data['ComplexClassification'] != 'blank') & (df_data['Months'] <= 12) & (df_data['AvgTimesMonth'] <= 31) & (df_data['MinutesOccasion'] != 9999) & (df_data['HipPain'] != 9999) & (df_data['ReducedTime']!=9999) & (df_data['HealthProblem']!=9999)]


# In[1076]:


df_data


# In[1077]:


# delete the different unit of 'MinutesOccasion' such as '120n' and '2.5 Hrs'
df_data = df_data[df_data['MinutesOccasion'].astype(str).str.isdigit()]
df_data


# In[1078]:


df_data['MinutesOccasion'] = df_data['MinutesOccasion'].astype(float)


# In[1079]:


# remove the wrong data
df_900 = df_900[((df_900['avgTimesMonth'] <= 31) & (df_900['HipPain'] != 9999) & (df_900['ReducedTime'] != 9999))]
df_900.head()


# In[1083]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df_data.groupby('HipPain').describe()


# In[1007]:


# Output the datasets
df_data.to_csv('df_data.csv',index=False)
df_900.to_csv('df_900.csv',index=False)


# # Find the realtionshiop and correlation

# In[1008]:


pain = df_data[df_data.HipPain==1]
print(pain.shape)


# In[1009]:


no_pain = df_data[df_data.HipPain==0]
print(no_pain.shape)


# In[1010]:


_, axes = plt.subplots(2,4, figsize=(25, 25))
ax = axes.ravel() # flatten the 2D array

for i in range(len(df_data.columns)):  # for each of all the features
    bins = 40

    #---plot histogram for each feature---
    ax[i].hist(pain.iloc[:,i], bins=bins, color='r', alpha=.5)
    ax[i].hist(no_pain.iloc[:,i], bins=bins, color='b', alpha=0.3)
    #---set the title---
    ax[i].set_title(df_data.columns[i], fontsize=12)    
    #---display the legend---
    ax[i].legend(['pain','no_pain'], loc='best', fontsize=8)
    
    bars = df_data.iloc[:,i].unique()
    x_pos = np.arange(len(bars))
    # Create names on the x axis
    plt.xticks(x_pos, bars)

    
plt.tight_layout()
plt.show()


# In[1011]:


df_data['ComplexClassification'].unique()


# In[1012]:


mapping2 = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6}
df_data = df_data.replace({'ComplexClassification': mapping2})
df_data.head()   


# In[1013]:


corr_data = df_data


# In[1014]:


# multiply columns of 'Months','AvgTimesMonth','MinutesOccasion' and siign the result to the new column
corr_data.insert(2,"Total", df_data['Months']*df_data['AvgTimesMonth']*df_data['MinutesOccasion'], True)
corr_data = corr_data.drop(["Months","AvgTimesMonth","MinutesOccasion"],axis=1)
corr_data.head()


# In[1015]:


corr_data["Total"] = corr_data["Total"].astype(int)


# In[1016]:


from sklearn.model_selection import train_test_split
import statsmodels.api as sm

temp = corr_data.drop(['HipPain'], axis=1)

X = temp   # independent variables
y = corr_data.loc[:,'HipPain']   # dependent variables

# Split the data into train and test with stratiefied sampling method
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100,stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[1017]:


corr_data.describe()


# In[1018]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(corr_data.corr(),annot = True, cmap='jet') 
# There is a strong positive correlation between HipPain and ReducedTime.


# In[1019]:


# get the correlation of each feature with respect to the target(HipPain)
df_corr = corr_data.corr()['HipPain'].abs().sort_values(ascending=False)
df_corr


# In[1020]:


# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot
ax = sns.countplot(x='ComplexClassification',data=corr_data,hue = 'HipPain')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# Display count on top of seaborn barplot
for container in ax.containers:
    ax.bar_label(container)


# Accuracy score

# In[1021]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[1022]:


from sklearn.metrics import mean_absolute_error
# make class predictions for the testing set
y_pred = logmodel.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)


# In[1023]:


# examine the class distribution of the testing set (using a Pandas Series method)
y_test.value_counts()


# In[1024]:


# calculate the percentage of ones
# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
y_test.mean()


# In[1025]:


# calculate the percentage of zeros
1 - y_test.mean()


# # Metrics computed from a confusion matrix (before thresholding)

# In[1026]:


# Confusion matrix is used to evaluate the correctness of a classification model
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# In[1027]:


TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]


# In[1028]:


TN


# In[1029]:


# Classification Accuracy: Overall, how often is the classifier correct?
# use float to perform true division, not integer division
print((TP + TN) / sum(map(sum, confusion_matrix)))
print(metrics.accuracy_score(y_test, y_pred))


# In[1030]:


# Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, y_pred))


# In[1031]:


# Specificity: When the actual value is negative, how often is the prediction correct?
specificity = TN / (TN + FP)
print(specificity)

from imblearn.metrics import specificity_score
specificity_score(y_test, y_pred)


# In[1032]:


# False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)


# In[1033]:


# Precision: When a positive value is predicted, how often is the prediction correct?
precision = TP / float(TP + FP)
print(precision)
print(metrics.precision_score(y_test, y_pred))


# In[1034]:


# F score
f_score = 2*TP / (2*TP + FP + FN)
print(f_score)
print(metrics.f1_score(y_test,y_pred))


# In[1035]:


#Evaluate the model using other performance metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[1036]:


from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = None)

cm_display.plot()
plt.show()


# # Adjusting the classification threshold¶

# In[1037]:


# store the predicted probabilities for class 1
y_pred_prob = logmodel.predict_proba(X_test)[:, 1]
y_pred_prob


# In[1061]:


# histogram of predicted probabilities
# 8 bins
plt.hist(y_pred_prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of HipPain')
plt.ylabel('Frequency')


# We can see from the bar:
# - The majority of observations have probability from 0.0 to 0.1
# - Small number of observations with probability > 0.4
# - This is below the threshold of 0.5
# - Most would be predicted "no HipPain" in this case
# 
# Solution:
# - Decrease the threshold for predicting HipPain
#     - Increase the sensitivity of the classifier
#         - This would increase the number of TP
#         - More sensitive to positive instances

# In[1039]:


# predict HitPain if the predicted probability is greater than 0.1
from sklearn.preprocessing import Binarizer
# it will return 1 for all values above 0.1 and 0 otherwise
# results are 2D so we slice out the first column
y_pred_class = Binarizer(threshold=0.1).transform(logmodel.predict_proba(X_test))[:,1]
y_pred_class


# In[1040]:


# print the first 10 predicted probabilities
y_pred_prob[0:10]


# In[1041]:


# print the first 10 predicted classes with the lower threshold
y_pred_class[0:10]


# In[1042]:


# previous confusion matrix (default threshold of 0.5)
print(matrix)


# In[1043]:


# new confusion matrix (threshold of 0.1)
new_confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(new_confusion)
# The row totals are the same
# The rows represent actual response values:
# 9489 values top row ; 2582 values bottom row

# Observations from the left column moving to the right column because we will have more TP and FP


# In[1044]:


TP = new_confusion[1, 1]
TN = new_confusion[0, 0]
FP = new_confusion[0, 1]
FN = new_confusion[1, 0]


# In[1045]:


# Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
sensitivity = TP / float(FN + TP)
print(round(sensitivity,2))
# sensitivity has increased (used to be 0.04)


# In[1046]:


# Specificity: When the actual value is negative, how often is the prediction correct?
specificity = TN / (TN + FP)
print(round(specificity,2))
# # specificity has decreased (used to be 1.00)


# In[1047]:


# F score
f_score = 2*TP / (2*TP + FP + FN)
print(f_score)
print(metrics.f1_score(y_test,y_pred_class))


# In[1048]:


#Evaluate the model using other performance metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class))


# Conclusion:
# 
# 1. Threshold of 0.5 is used by default (for binary problems) to convert predicted probabilities into class predictions
# 2. Threshold can be adjusted to increase sensitivity or specificity
# 3. Sensitivity and specificity have an inverse relationship
# - Increasing one would always decrease the other
# 4. Adjusting the threshold should be one of the last step you do in the model-building process
# - The most important steps are:
#     - Building the models
#     - Selecting the best model

# # Obtain Optimal Probability Thresholds with ROC Curve 

# Question: Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
# 
# Answer: Plot the Receiver Operating Characteristic (ROC) curve.

# In[1049]:


# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot([0,1], [0,1], linestyle="--") # plot random curve
plt.plot(fpr, tpr,marker=".")
plt.title('ROC curve for HipPain classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[1050]:


from numpy import argmax

# get the best threshold with Youden’s J statistic: J = TruePositiveRate – FalsePositiveRate
J = tpr - fpr
ix = argmax(J)
best_threshold = thresholds[ix]
print('Best Threshold=%f' % (best_threshold))


# In[1051]:


roc_predictions = [1 if i >= best_threshold else 0 for i in y_pred_prob]


# # Evaluate Model (After Thresholding)

# In[1052]:


print(f"Accuracy Score Before and After Thresholding: {round(metrics.accuracy_score(y_test, y_pred),2)}, {round(metrics.accuracy_score(y_test, roc_predictions),2)}")
print(f"Precision Score Before and After Thresholding: {round(metrics.precision_score(y_test, y_pred),2)}, {round(metrics.precision_score(y_test, roc_predictions),2)}")
print(f"Recall Score Before and After Thresholding: {round(metrics.recall_score(y_test, y_pred),2)}, {round(metrics.recall_score(y_test, roc_predictions),2)}")
print(f"Specificity Score Before and After Thresholding: {round(specificity_score(y_test, y_pred),2)}, {round(specificity_score(y_test, roc_predictions),2)}")
print(f"F1 Score Before and After Thresholding: {round(metrics.f1_score(y_test, y_pred),2)}, {round(metrics.f1_score(y_test, roc_predictions),2)}")


# In[1053]:


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[1054]:


evaluate_threshold(0.5)

