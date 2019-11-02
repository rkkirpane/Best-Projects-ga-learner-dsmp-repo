# --------------
# Importing header files
import numpy as np
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path, delimiter="," , skip_header=1)
#New record

print(data.shape)
census=np.concatenate((data,new_record),axis=0)
print(census.shape)
#Code starts here



# --------------
#Code starts here
age=census[:,0]
max_age=age.max()
min_age=age.min()
age_mean=age.mean()
age_std=np.std(age)
print(max_age)
print(min_age)
print(age_mean)
print(age_std)


# --------------
race = census[:, 2].astype(int)
print(race)

race_0=census[race==0]
race_1=census[race==1]
race_2=census[race==2]
race_3= census[race==3]
race_4 =census[race==4]

#print([race_0, race_1, race_2, race_3, race_4])

len_0=len(race_0)
print(len_0)
len_1=len(race_1)
print(len_1)
len_2=len(race_2)
print(len_2)
len_3=len(race_3)
print(len_3)
len_4=len(race_4)
print(len_4)

race_list=[len_0,len_1,len_2,len_3,len_4]

minority_race=race_list.index(min(race_list))

print(minority_race)




# --------------
#Code starts here
senior_citizens=census[age>60]
#print(senior_citizens)
working_hours_sum=np.sum(senior_citizens[:,6])
print(working_hours_sum)
senior_citizens_len=len(senior_citizens)
print(senior_citizens_len)
avg_working_hours=(working_hours_sum/senior_citizens_len)
print(avg_working_hours)


# --------------
#Code starts here
high=census[census[:,1] > 10]
print(high)
low=census[census[:,1] <= 10]
print(low)
avg_pay_high=np.mean(high[:,7])
print(avg_pay_high)
avg_pay_low=np.mean(low[:,7])
print(avg_pay_low)


