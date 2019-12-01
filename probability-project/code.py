# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(path)
df.head(5)
p_a=len(df[df.fico > 700])/len(df.fico)
print(p_a)
p_b=len(df[df.purpose == 'debt_consolidation'])/len(df.purpose)
print(p_b)
df1=df[df.purpose == 'debt_consolidation']
#df1
p_a_b=df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
print(p_a_b)
result=(p_a_b==p_a)
print(result)

# code ends here


# --------------
# code starts here
prob_lp=len(df[df['paid.back.loan']=='Yes'])/df['paid.back.loan'].shape[0]
prob_cs=len(df[df['credit.policy']== 'Yes'])/df['credit.policy'].shape[0]
new_df=df[df['paid.back.loan'] == 'Yes']
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]
bayes=(prob_pd_cs * prob_lp)/ prob_cs
print(bayes)




# code ends here


# --------------
# code starts here


df['purpose'].value_counts().plot(kind='bar')
df1=df[df['paid.back.loan']=='No']
df1['purpose'].value_counts().plot(kind='bar')


# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
plt.hist(df['installment'])
plt.hist(df['log.annual.inc'])



# code ends here


