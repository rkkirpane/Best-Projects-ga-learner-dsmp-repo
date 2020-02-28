# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers=pd.read_excel(path, sheet_name=0)

# Load Transactions
transactions=pd.read_excel(path, sheet_name=1)
#print(offers.head())
#print(transactions.head())
# Merge dataframes
transactions['n']=1
#print(transactions.head())
df=pd.merge(offers, transactions, how='left')
print(df.head(5))
# Look at the first 5 rows



# --------------
# Code starts here

# create pivot table
matrix = pd.pivot_table(df, index='Customer Last Name', columns='Offer #', values='n')

# replace missing values with 0
matrix.fillna(0,inplace=True)
# reindex pivot table
matrix.reset_index(inplace=True)
# display first 5 rows
print(matrix.head(5))

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,[0, 33, 34, 35]]

# visualize clusters
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data=pd.merge(clusters,transactions)

# merge `data` and `offers`
data=pd.merge(offers,data)
# initialzie empty dictionary
champagne={}

# iterate over every cluster
for i in range(0,5):
    new_df=data[data['cluster']==i]
    counts=new_df['Varietal'].value_counts(ascending=False)
    if counts.index[0]=='Champagne':
        champagne[i]=(counts[0])
cluster_champagne=max(champagne,key=champagne.get)
print(cluster_champagne)
    # observation falls in that cluster

    # sort cluster according to type of 'Varietal'

    # check if 'Champagne' is ordered mostly

        # add it to 'champagne'


# get cluster with maximum orders of 'Champagne' 


# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount={}


# iterate over every cluster
for i in range(0,5):
    new_df=data[data["cluster"]==i]
    counts=sum(new_df["Discount (%)"])/len(new_df)
    discount.update({i:counts})
# Code ends here
cluster_discount=max(discount,key=discount.get)
print(cluster_discount)


