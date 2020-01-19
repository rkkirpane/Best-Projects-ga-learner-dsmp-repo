# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df=pd.read_csv(path)
# Code starts here
X=df.drop(['customerID','Churn'],axis=1)
y=df.Churn
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0) 



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder
X_train.TotalCharges=X_train.TotalCharges.replace('',np.NaN)
X_train.TotalCharges=X_train.TotalCharges.convert_objects(convert_numeric=True)
X_test.TotalCharges=X_test.TotalCharges.convert_objects(convert_numeric=True)
X_train.TotalCharges.isna().sum()
X_train.TotalCharges=X_train.TotalCharges.fillna(X_train.TotalCharges.mean())
X_test.TotalCharges=X_test.TotalCharges.fillna(X_test.TotalCharges.mean())
X_test.TotalCharges.isna().sum()
X_train.TotalCharges.isna().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
categorical_feature_mask = X_train.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols_X_train = X_train.columns[categorical_feature_mask].tolist()
print(categorical_cols_X_train)
categorical_feature_mask_X_test = X_test.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols_X_test = X_test.columns[categorical_feature_mask_X_test].tolist()
print(categorical_cols_X_test)
X_train[categorical_cols_X_train] = X[categorical_cols_X_train].apply(lambda col: le.fit_transform(col))
X_train[categorical_cols_X_train].head(10)
X_test[categorical_cols_X_test] = X[categorical_cols_X_test].apply(lambda col: le.fit_transform(col))
X_test[categorical_cols_X_test].head(10)
y_train=y_train.replace({'No':0, 'Yes':1})
y_test=y_test.replace({'No':0, 'Yes':1})


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
ada_model=AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred=ada_model.predict(X_test)
ada_score=accuracy_score(y_test,y_pred)
print('ada_score:',ada_score)
ada_cm=confusion_matrix(y_test,y_pred)
print("ada_cm:",ada_cm)
ada_cr=classification_report(y_test,y_pred)
print("ada_cr:",ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model=XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
xgb_score=accuracy_score(y_test,y_pred)
print("xgb_score:",xgb_score)
xgb_cm=confusion_matrix(y_test,y_pred)
print("xgb_cm:",xgb_cm)
xgb_cr=classification_report(y_test,y_pred)
print("xgb_cr:",xgb_cr)

clf_model=GridSearchCV(estimator=xgb_model , param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred=clf_model.predict(X_test)
clf_score=accuracy_score(y_test,y_pred)
print("clf_score:",clf_score)
clf_cm=confusion_matrix(y_test,y_pred)
print("clf_cm:",clf_cm)
clf_cr=classification_report(y_test,y_pred)
print("clf_cr:",clf_cr)







