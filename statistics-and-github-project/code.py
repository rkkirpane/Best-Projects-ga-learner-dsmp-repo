# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
data.Gender.replace('-','Agender',inplace=True)
gender_count=data.Gender.value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment=data.Alignment.value_counts()
alignment.plot(kind='pie')
plt.title('Character Alignment')



# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
sc_covariance=data[['Strength', 'Combat']].cov().iloc[0,1]
#sc_covariance
sc_strength=data.Strength.std()
#sc_strength
sc_combat=data.Combat.std()
#sc_combat
sc_pearson=sc_covariance/((sc_strength)*(sc_combat))
print(sc_pearson)

ic_df=data[['Intelligence','Combat']]
ic_covariance=data[['Intelligence', 'Combat']].cov().iloc[0,1]
#ic_covariance
ic_intelligence=data.Intelligence.std()
#ic_Intelligence
ic_combat=data.Combat.std()
#ic_combat
ic_pearson=ic_covariance/((ic_intelligence)*(ic_combat))
print(ic_pearson)


# --------------
#Code starts here
total_high=data.Total.quantile(q=0.99)
#total_high
super_best=data[data['Total']> total_high]
#super_best
super_best_names=[super_best['Name']]
print(super_best_names)


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(10,10))
ax_1.boxplot(data.Intelligence)
ax_1.set_title("Intelligence")
ax_2.boxplot(data.Speed )
ax_2.set_title("Speed")
ax_2.boxplot(data.Power) 
ax_3.set_title("Power")
plt.show()


