# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data=pd.read_csv(path)
data.hist('Rating',bins=5)
data=data[data['Rating']<=5]
data.hist('Rating',bins=5)

#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
percent_null= total_null/data.isnull().count()
print(total_null,percent_null)
missing_data=pd.concat([total_null,percent_null],keys=['Total','Percent'], axis=1)
print('missing_data') 
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
print(total_null,percent_null)
missing_data_1 = pd.concat([total_null_1, percent_null_1], keys=['Total', 'Percent'],axis = 1)
print('missing_data_1') 


# --------------

#Code starts here
sns.catplot(x="Category", y="Rating", data=data,kind="box",height = 10)
import matplotlib.pyplot as plt
plt.xticks(rotation = 90)
plt.title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']= data['Installs'].astype('int64')
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
sns.regplot(x="Installs",y="Rating",data = data)
plt.title('Rating vs Installs [RegPlot]')



# --------------
#Code starts here
data['Price'].value_counts()
data['Price'] = data['Price'].str.replace('$','')
data['Price']= data['Price'].astype(float)
sns.regplot(x="Price",y="Rating",data = data)
plt.title('Rating vs Price [RegPlot]')







#Code ends here


# --------------

#Code starts here

data.Genres.unique()
new = data["Genres"].str.split(pat = ';',expand = True)
data["Genres"] = new[0]
gr_mean= data.groupby('Genres',as_index=False)[['Rating']].mean()
gr_mean.describe()
gr_mean=gr_mean.sort_values('Rating',ascending=True)
print(gr_mean.head(1))
print(gr_mean.tail(1))






# --------------

#Code starts here
data['Last Updated']
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date-data['Last Updated']).dt.days
sns.regplot(x="Last Updated Days",y="Rating",data = data)
plt.title('Rating vs Price [RegPlot]')




#Code ends here


