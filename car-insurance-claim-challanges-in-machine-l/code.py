# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(path)
df.head(5)
df.info()
df.INCOME=df.INCOME.str.replace('$', '')
df.INCOME=df.INCOME.str.replace(',', '')
df['HOME_VAL']=df['HOME_VAL'].str.replace('$', '')
df['HOME_VAL']=df['HOME_VAL'].str.replace(',', '')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace('$', '')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace(',', '')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace('$', '')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace(',', '')
df['CLM_AMT']=df['CLM_AMT'].str.replace('$', '')
df['CLM_AMT']=df['CLM_AMT'].str.replace(',', '')
X=df.drop("CLAIM_FLAG",1)
y=df.CLAIM_FLAG 

count=df.CLAIM_FLAG.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6)  



# Code ends here


# --------------
# Code starts here
X_train.INCOME=X_train.INCOME.astype('float')
X_train.HOME_VAL=X_train.HOME_VAL.astype('float')
X_train.BLUEBOOK=X_train.BLUEBOOK.astype('float')
X_train.OLDCLAIM=X_train.OLDCLAIM.astype('float')
X_train.CLM_AMT=X_train.CLM_AMT.astype('float')
X_test.INCOME=X_test.INCOME.astype('float')
X_test.HOME_VAL=X_test.HOME_VAL.astype('float')
X_test.BLUEBOOK=X_test.BLUEBOOK.astype('float')
X_test.OLDCLAIM=X_test.OLDCLAIM.astype('float')
X_test.CLM_AMT=X_test.CLM_AMT.astype('float')
X_train.isna().sum()
X_test.isna().sum()










# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
# Code ends here
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
y_train=y_train[X_train.index] 
y_test=y_test[X_test.index] 

X_train["AGE"]=X_train.AGE.fillna(X_train.AGE.mean(),inplace=True)
X_train["CAR_AGE"]=X_train.CAR_AGE.fillna(X_train.CAR_AGE.mean(),inplace=True)
X_train["INCOME"]=X_train.INCOME.fillna(X_train.INCOME.mean(),inplace=True)
X_train["HOME_VAL"]=X_train.HOME_VAL.fillna(X_train.HOME_VAL.mean(),inplace=True)
X_test["AGE"]=X_test.AGE.fillna(X_train.AGE.mean(),inplace=True)
X_test["CAR_AGE"]=X_test.CAR_AGE.fillna(X_train.CAR_AGE.mean(),inplace=True)
X_test["INCOME"]=X_test.INCOME.fillna(X_train.INCOME.mean(),inplace=True)
X_test["HOME_VAL"]=X_test.HOME_VAL.fillna(X_train.HOME_VAL.mean(),inplace=True)








# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le=LabelEncoder()
for i in columns:
    X_train[i]=le.fit_transform(X_train[i])
    X_test[i]=le.transform(X_test[i])


# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#X_train.dropna(inplace=True)
#y_train.dropna(inplace=True)
# code starts here 
model=LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print("score :",score)



# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=9)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)
# Code ends here


