"Problem 2"

import pandas as pd

# Read data into Python
heartdisease = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis/heart disease.csv")
heartdisease.head()
heartdisease.shape
heartdisease.columns.values
heartdisease.dtypes
heartdisease.info()

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
heartdisease.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
heartdisease.mean()
heartdisease.median()
heartdisease.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
heartdisease.var() 
heartdisease.std()

#3rd moment Business Decision
heartdisease.skew()

#4th moment Business Decision
heartdisease.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(heartdisease):
    plt.figure(i)
    sns.histplot(data=heartdisease, x=predictor)
    
#boxplot    
for i, predictor in enumerate(heartdisease):
    plt.figure(i)
    sns.boxplot(data=heartdisease, x=predictor)

###################### Outlier Treatment #########
# only for numerical data - which has outliers
'trestbps','chol','thalach','oldpeak'
# so we have 4 variables which has outlierswe leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)

# let's find outliers 
"trestbps"
sns.boxplot(heartdisease['trestbps']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['trestbps'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
heartdisease_t = winsorizer.fit_transform(heartdisease[['trestbps']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(heartdisease_t['trestbps']);plt.title('trestbps');plt.show()

#we see no outiers

"chol"
sns.boxplot(heartdisease['chol']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['chol'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
heartdisease_t = winsorizer.fit_transform(heartdisease[['chol']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(heartdisease_t['chol']);plt.title('chol');plt.show()

#we see no outiers

"thalach"
sns.boxplot(heartdisease['thalach']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['thalach'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
heartdisease_t = winsorizer.fit_transform(heartdisease[['thalach']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(heartdisease_t['thalach']);plt.title('thalach');plt.show()

#we see no outiers

"oldpeak"
sns.boxplot(heartdisease['oldpeak']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['oldpeak'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
heartdisease_t = winsorizer.fit_transform(heartdisease[['oldpeak']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(heartdisease_t['oldpeak']);plt.title('oldpeak');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
heartdisease.isna().sum()

# there is no na values

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed - numeric data not binary or categorical
"age"
stats.probplot(heartdisease.age, dist='norm',plot=pylab) #pylab is visual representation
"trestbps"
stats.probplot(heartdisease.trestbps, dist='norm',plot=pylab)
#transformation to make trestbps variable normal
stats.probplot(np.sqrt(heartdisease.trestbps),dist="norm",plot=pylab)
"chol"
stats.probplot(heartdisease.chol, dist='norm',plot=pylab)
"thalach"
stats.probplot(heartdisease.thalach, dist='norm',plot=pylab)
"oldpeak"
stats.probplot(heartdisease.oldpeak, dist='norm',plot=pylab)
stats.probplot(np.log10(heartdisease.oldpeak),dist="norm",plot=pylab)

"5.3 Perform clustering before and after applying PCA to cross the number of clusters "
"Hierarchical Chustering"
# Normalization function using z std. all are continuous data.
 
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
# Normalized data frame (considering the numerical part of data)
heartdisease_norm = norm_func(heartdisease) # we take numeric columns, becuase that binary varibales create problem while clustering
heartdisease.describe() # min=0, max=1
heartdisease_norm.info()

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(heartdisease_norm, method = "complete", metric = "euclidean")
z

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying Agglomerative Clustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(heartdisease_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
heartdisease.info()
heartdisease['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
heartdisease1 = heartdisease.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]] 
heartdisease1.head()

# Aggregate mean of each cluster
heartdisease.iloc[:].groupby(heartdisease1.clust).mean() # from sat it will calculate

# creating a csv file
heartdisease1.to_csv("heartdiseaseHierarchicaloutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"Before PCA we will find the pattern of the Patient using Hierarchical clustering"

"Patient Pattern 1 : 1st group" 
"Patient Pattern 2 : 2nd group"
"Patient Pattern 3 : 0th group"
"Patient age of 62 has high risk of heart disease after that 56 age less compared to 62 age and last 54"

"Now let's try wth K-Means Clustering"

from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 9)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heartdisease_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(heartdisease_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
heartdisease['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
heartdisease2 = heartdisease.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
heartdisease2.head()
heartdisease_norm.head()

heartdisease2.iloc[:,1:].groupby(heartdisease.clust).mean()

heartdisease2.to_csv("heartdiseaseKMeansoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"Before PCA we will find the pattern of the Patient using Hierarchical clustering "

"Patient Pattern 1 : 1st group" 
"Patient Pattern 2 : 2nd group"
"Patient Pattern 3 : 0th group"

"Patient age of 62 has high risk of heart disease after that 56 age less compared to 62 age and last 54"

#Model building
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Normalizing the numeric data
heartdisease_normal1 = scale(heartdisease)
heartdisease_normal1

pca = PCA(n_components = 14) 
#  idealy we pass equal number of columns
pca_values = pca.fit_transform(heartdisease_normal1) # pca is done, transform value into pca
# pca values gives scores 
#The amount of variance that each PCA explains is
var = pca.explained_variance_ratio_ # how much information they capture in percentage (%)
var

pca.components_ # weights
pca.components_[0] # if we select 1st pc

# Cumulatie Variance
var1 = np.cumsum(np.round(var, decimals = 4) * 100) # choose only 4 decimals
var1 # it converts variance into percentage

# Variance plot for PCA compnents obtained
plt.plot(var1, color = "red") # by seeing the plot we can also say how many columns we need to choose

#PCA scores
pca_values

# make them as data frame
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "pc0","pc1","pc2","pc03","pc04","pc05",'pc06','pc07','pc08','pc09','pc10','pc11','pc12','pv13'

final = pd.concat([heartdisease['target'],pca_data.iloc[:, 0:3]], axis = 1) 

#Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.pc0, y = final.pc1)

"Hirarchical and K-Means after applying PCA"

"Hierarchical Chustering"

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(heartdisease_normal1, method = "complete", metric = "euclidean")
z

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying Agglomerative Clustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(heartdisease_normal1)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
final.info()
final['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
final = final.iloc[:,[4,1,2,3]] # we take numeric columns, becuase that binary varibales create problem while clustering
final.head()

# Aggregate mean of each cluster
final.iloc[:,1:].groupby(final.clust).mean() # from sat it will calculate

# creating a csv file
final.to_csv("heartdiseaseHierarchicalPCAoutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"After PCA we will find the pattern of the Patient using Hierarchical clustering "

"Patient Pattern 1 : 1st group" 
"Patient Pattern 2 : 2nd group"
"Patient Pattern 3 : 0th group"

"Now let's try wth K-Means Clustering"

from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 15)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heartdisease_normal1)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(heartdisease_normal1)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
final['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
final = final.iloc[:,[4,1,2,3]]
final.head()

final.iloc[:,1:].groupby(final.clust).mean()

final.to_csv("heartdiseaseKMeansPCAoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"After PCA we will find the pattern of the Patient using Hierarchical clustering "

"Patient Pattern 1 : 1st group" 
"Patient Pattern 2 : 2nd group"
"Patient Pattern 3 : 0th group"



