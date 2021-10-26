"Problem 1"

import pandas as pd

# Read data into Python
wine = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis/wine.csv")
wine.head()
wine.shape
wine.columns.values
wine.dtypes
wine.info()

#Considering only useful columns
wine.data = wine.iloc[:,1:] # it's not data frame it's only capture the data
type(wine.data)

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
wine.data.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
wine.data.mean()
wine.data.median()
wine.data.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
wine.data.var() 
wine.data.std()

#3rd moment Business Decision
wine.data.skew()

#4th moment Business Decision
wine.data.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(wine.data):
    plt.figure(i)
    sns.histplot(data=wine.data, x=predictor)
    
#boxplot    
for i, predictor in enumerate(wine.data):
    plt.figure(i)
    sns.boxplot(data=wine.data, x=predictor)

###################### Outlier Treatment #########
# only for numerical data - which has outliers
'Malic','Ash','Alcalinity','Magnesium','Proanthocyanins','Color','Hue'
# so we have 7 variables which has outlierswe leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)

# let's find outliers 
"Malic"
sns.boxplot(wine.data['Malic']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Malic'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Malic']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Malic']);plt.title('Malic');plt.show()

#we see no outiers

"Ash"
sns.boxplot(wine.data['Ash']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Ash'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Ash']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Ash']);plt.title('Ash');plt.show()

#we see no outiers

"Alcalinity"
sns.boxplot(wine.data['Alcalinity']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Alcalinity'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Alcalinity']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Alcalinity']);plt.title('Alcalinity');plt.show()

#we see no outiers

"Magnesium"
sns.boxplot(wine.data['Magnesium']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Magnesium'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Magnesium']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Magnesium']);plt.title('Magnesium');plt.show()

#we see no outiers

"Proanthocyanins"
sns.boxplot(wine.data['Proanthocyanins']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Proanthocyanins'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Proanthocyanins']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Proanthocyanins']);plt.title('Proanthocyanins');plt.show()

#we see no outiers

"Color"
sns.boxplot(wine.data['Color']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Color'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Color']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Color']);plt.title('Color');plt.show()

#we see no outiers

"Hue"
sns.boxplot(wine.data['Hue']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Hue'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
wine.data_t = winsorizer.fit_transform(wine.data[['Hue']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(wine.data_t['Hue']);plt.title('Hue');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
wine.data.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = wine.data.duplicated()
sum(duplicate)

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
"Alcohol"
stats.probplot(wine.data.Alcohol, dist='norm',plot=pylab) #pylab is visual representation
"Malic"
stats.probplot(wine.data.Malic, dist='norm',plot=pylab)

#transformation to make workex variable normal
import numpy as np
stats.probplot(np.log2(wine.data.Malic),dist="norm",plot=pylab) #best transformation
"Ash"
stats.probplot(wine.data.Ash, dist='norm',plot=pylab)
"Alcalinity"
stats.probplot(wine.data.Alcalinity, dist='norm',plot=pylab)
"Magnesium"
stats.probplot(wine.data.Magnesium, dist='norm',plot=pylab)
stats.probplot(np.log2(wine.data.Magnesium),dist="norm",plot=pylab)
"Phenols"
stats.probplot(wine.data.Phenols, dist='norm',plot=pylab)
"Flavanoids"
stats.probplot(wine.data.Flavanoids, dist='norm',plot=pylab)
stats.probplot(wine.data.Flavanoids*wine.data.Flavanoids,dist="norm",plot=pylab)
"Nonflavanoids"
stats.probplot(wine.data.Nonflavanoids, dist='norm',plot=pylab)
stats.probplot(np.log2(wine.data.Nonflavanoids),dist="norm",plot=pylab)
"Proanthocyanins"
stats.probplot(wine.data.Proanthocyanins, dist='norm',plot=pylab)
"Color"
stats.probplot(wine.data.Color, dist='norm',plot=pylab)
stats.probplot(np.log2(wine.data.Color),dist="norm",plot=pylab)
"Hue"
stats.probplot(wine.data.Hue, dist='norm',plot=pylab)
"Dilution"
stats.probplot(wine.data.Dilution, dist='norm',plot=pylab)
stats.probplot(wine.data.Dilution*wine.data.Dilution,dist="norm",plot=pylab)
"Proline"
stats.probplot(wine.data.Proline, dist='norm',plot=pylab)
stats.probplot(np.log2(wine.data.Proline),dist="norm",plot=pylab)

"5.3 Perform clustering before and after applying PCA to cross the number of clusters "
"Hierarchical Chustering"
# Normalization function using z std. all are continuous data.
 
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
# Normalized data frame (considering the numerical part of data)
wine.data_norm = norm_func(wine.data) # we take numeric columns, because that binary varibales create problem while clustering
wine.data.describe() # min=0, max=1
wine.data_norm.info()

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(wine.data_norm, method = "complete", metric = "euclidean")
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

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(wine.data_norm)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
wine.data.info()
wine['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
wine1 = wine.iloc[:,[14,1,2,3,4,5,6,7,8,9,10,11,12,13]] # we take numeric columns, becuase that binary varibales create problem while clustering
wine1.head()

# Aggregate mean of each cluster
wine.iloc[:,1:].groupby(wine1.clust).mean() # from sat it will calculate

# creating a csv file
wine1.to_csv("wineHierarchicaloutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"Before PCA we will find the pattern using Hierarchical clustering the Quality of wine "

"Wine Quality 1 : 2nd group"
"Wine Quality 2 : 1st group"
"Wine Quality 3 : 0th group"

"Now let's try wth K-Means Clustering"

from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 9)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine.data_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine.data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
wine['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
wine2 = wine.iloc[:,[14,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine2.head()
wine.data_norm.head()

wine2.iloc[:,1:].groupby(wine.clust).mean()

wine2.to_csv("wineKMeansoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"Before PCA we will find the pattern using K-Means clustering the Quality of wine "

"Wine Quality 1 : 2nd group"
"Wine Quality 2 : 1st group"
"Wine Quality 3 : 0th group"

#Model building
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Normalizing the numeric data
wine.data1_normal = scale(wine.data)
wine.data1_normal

pca = PCA(n_components = 13) 
#  idealy we pass equal number of columns
pca_values = pca.fit_transform(wine.data1_normal) # pca is done, transform value into pca
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
pca_data.columns = "pc0","pc1","pc2","pc03","pc04","pc05",'pc06','pc07','pc08','pc09','pc10','pc11','pc12'

final = pd.concat([wine['Type'], pca_data.iloc[:, 0:3]], axis = 1) 

#Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.pc0, y = final.pc1)

"Hirarchical and K-Means after applying PCA"

"Hierarchical Chustering"

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram

z = linkage(wine.data1_normal, method = "complete", metric = "euclidean")
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

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(wine.data1_normal)
h_complete.labels_ # labels gives cluster Id's

cluster_labels = pd.Series(h_complete.labels_) # convert into series (proper arranging the numbers)
final.info()
final['clust'] = cluster_labels #that cluster_labels which name as 'clust' is added in last column
final = final.iloc[:,[4,1,2,3]] # we take numeric columns, becuase that binary varibales create problem while clustering
final.head()

# Aggregate mean of each cluster
final.iloc[:,1:].groupby(final.clust).mean() # from sat it will calculate

# creating a csv file
final.to_csv("wineHierarchicalPCAoutput1python.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"After PCA we will find the pattern using Hierarchical clustering the Quality of wine "

"Wine Quality 1 : 1st group"
"Wine Quality 2 : 0th group"
"Wine Quality 3 : 2nd group"

"Now let's try wth K-Means Clustering"

from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 15)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine.data1_normal)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine.data1_normal)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
final['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
final = final.iloc[:,[4,1,2,3]]
final.head()

final.iloc[:,1:].groupby(final.clust).mean()

final.to_csv("wineKMeansPCAoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Principal Component Analysis"
os.chdir(path) # current working directory

"After PCA we will find the pattern using K-Means clustering the Quality of wine "

"Wine Quality 1 : 2nd group"
"Wine Quality 2 : 0th group"
"Wine Quality 3 : 1st group"

