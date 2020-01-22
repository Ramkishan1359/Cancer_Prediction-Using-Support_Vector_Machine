import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

print(os.getcwd())

os.chdir('E:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')
dataset = pd.read_csv('Universities.csv')
dataset

X = dataset.iloc[:, [1,5]].values
X

km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)

y_km = km.fit_predict(X)
y_km

#Visualizing the clusters
plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',label='Cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='blue',label='Cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='green',label='Cluster 3')
plt.scatter(X[y_km==3,0],X[y_km==3,1],c='cyan',label='Cluster 4')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Universities')
plt.legend()
plt.show()

plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',label='Target First')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='blue',label='Standard Scores')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='green',label='Careful High Expenses')
plt.scatter(X[y_km==3,0],X[y_km==3,1],c='cyan',label='Target Next')
plt.title('Clusters of Universities')
plt.legend()
plt.show()

s1_grps = pd.Series(km.labels_)
s2_univs = dataset.iloc[:,0]
rslt = pd.concat([s1_grps,s2_univs],axis=1)
rslt

km.cluster_centers_
df = pd.DataFrame(km.cluster_centers_)
df.columns = ('Sat Score','Expenses')
df

s1_grps = pd.Series(km.labels_)
s2_univs = dataset.iloc[:,:]
rslt = pd.concat([s1_grps,s2_univs],axis=1)

rslt


X = dataset.iloc[:, [1,2,3,4,5,6]].values
X


km = KMeans(n_clusters=5, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',label='Cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='blue',label='Cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='green',label='Cluster 3')
plt.scatter(X[y_km==3,0],X[y_km==3,1],c='cyan',label='Cluster 4')
plt.scatter(X[y_km==4,0],X[y_km==4,1],c='pink',label='Cluster 5')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Universities')
plt.legend()
plt.show()

centers = pd.DataFrame(km.cluster_centers_)

centers["clusters"] = range(5) 
dataset["ind"] = dataset.index
dataset = dataset.merge(centers)
dataset.sample(5)

dataset = dataset.sort_values("ind")
dataset = dataset.drop("ind",1)
dataset

#Visualizing the clusters
plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',label='Target')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='blue',label='Top Level 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='green',label='IITs')
plt.scatter(X[y_km==3,0],X[y_km==3,1],c='cyan',label='Standard')
plt.scatter(X[y_km==4,0],X[y_km==4,1],c='pink',label='Above Average')
plt.title('Clusters of Universities')
plt.legend()
plt.show()

dataset = pd.read_csv('Universities.csv')
X = dataset.iloc[:, [1,5]].values
km = KMeans(n_clusters=8, init='k-means++',max_iter=300,n_init=10,random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0],X[y_km==0,1],c='red',label='Cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='blue',label='Cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],c='green',label='Cluster 3')
plt.scatter(X[y_km==3,0],X[y_km==3,1],c='cyan',label='Cluster 4')
plt.scatter(X[y_km==4,0],X[y_km==4,1],c='black',label='Cluster 5')
plt.scatter(X[y_km==5,0],X[y_km==5,1],c='orange',label='Cluster 6')
plt.scatter(X[y_km==6,0],X[y_km==6,1],c='pink',label='Cluster 7')
plt.scatter(X[y_km==7,0],X[y_km==7,1],c='indigo',label='Cluster 8')

plt.title('Clusters of Universities')
plt.legend()
plt.show()