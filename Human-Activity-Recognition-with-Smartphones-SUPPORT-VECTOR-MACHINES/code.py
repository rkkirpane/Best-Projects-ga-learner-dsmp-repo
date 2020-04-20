# --------------
import pandas as pd
from collections import Counter

# Load dataset
data=pd.read_csv(path)
print(data.isna().sum())
print(data.describe())


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label=data.Activity
x=label
# plot the countplot
sns.countplot(x=label)


# --------------
# make the copy of dataset
# make the copy of dataset
data_copy = data.copy()

# Create an empty column 
data_copy['duration'] = ''

# Calculate the duration
duration_df = (data_copy.groupby([label[label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
plot_data = duration_df.reset_index().sort_values('duration', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})


# Plot the durations for staircase use
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()


# --------------
#exclude the Activity column and the subject column



#Calculate the correlation values


#stack the data and convert to a dataframe



#create an abs_correlation column



#Picking most correlated features without having self correlated pairs
feature_cols = data_copy.columns[: -3]   #exclude the Activity column
feature_cols
#Calculate the correlation values
correlated_values = data_copy[feature_cols].corr()
#stack the data and convert to a dataframe

correlated_values = (correlated_values.stack().to_frame().reset_index().rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'}))
correlated_values.head()
correlated_values['abs_correlation'] = correlated_values.Correlation_score.abs()
correlated_values.head()
s_corr_list = correlated_values.sort_values('abs_correlation', ascending = False)
print(s_corr_list.sample(5))
top_corr_fields=s_corr_list.query('abs_correlation>0.8')
top_corr_fields = top_corr_fields[top_corr_fields['Feature_1'] != top_corr_fields['Feature_2']].reset_index(drop=True) 

print(top_corr_fields)












# --------------
# importing neccessary libraries
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Activity']= le.fit_transform(data['Activity']) 




# Encoding the target variable



# split the dataset into train and test
X=data.drop("Activity",1)
y=data.Activity
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
# Baseline model 
from sklearn.svm import SVC
classifier= SVC()
clf=classifier.fit(X_train , y_train )
y_pred=clf.predict(X_test)
from sklearn.metrics import precision_score,accuracy_score
precision=precision_score(y_test, y_pred,average = 'weighted')
print("precision",precision)
accuracy =accuracy_score(y_test, y_pred)
print("accuracy",accuracy)
from sklearn.metrics import f1_score
f_score  =f1_score(y_test, y_pred,average = 'weighted')
print("f1_score",f1_score)
model1_score=accuracy_score(y_test, y_pred)
print("model1_score",model1_score)



# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC

# Feature selection using Linear SVC
lsvc=LinearSVC(penalty='l1', dual=False, C=0.01, random_state=42).fit(X_train,y_train)
model_2=SelectFromModel(lsvc,prefit=True)

new_train_features=model_2.transform(X_train)
new_test_features=model_2.transform (X_test)



# model building on reduced set of features
classifier_2= SVC()
clf_2=classifier_2.fit(new_train_features,y_train)
y_pred_new=clf_2.predict(new_test_features)
model2_score=accuracy_score(y_test, y_pred_new)
print(model2_score)

#precision=precision_score(y_test, y_pred_new)
#print("precision",precision)

from sklearn.metrics import f1_score
f_score  =f1_score(y_test, y_pred_new,average = 'weighted')
print("f_score",f_score)


# --------------
# Importing Libraries
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters

parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [100, 20, 1, 0.1]
}

# Usage of grid search to select the best hyperparmeters
selector = GridSearchCV(SVC(), parameters, scoring='accuracy') 
selector.fit(new_train_features, y_train)

print('Best parameter set found:')
print(selector.best_params_)
print('Detailed grid scores:')
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, selector.cv_results_['params']):
    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()
    

# Model building after Hyperparameter tuning
classifier_3 = SVC(kernel='rbf', C=100)
clf_3 = classifier_3.fit(new_train_features, y_train)
y_pred_final = clf_3.predict(new_test_features)
model3_score = accuracy_score(y_test, y_pred_final)

print('Accuracy score:', model3_score)





