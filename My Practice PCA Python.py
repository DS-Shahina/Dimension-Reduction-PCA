import pandas as pd
import numpy as np

univ1 = pd.read_excel("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Principal Component Analysis/University_Clustering.xlsx")
univ1.describe()
univ1.info() #data types , and missing values also

univ = univ1.drop(["State"], axis = 1) #it changes into original dataframe

# function check in the package - dir(pd)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Considering only numeric data
univ.data = univ.iloc[:,1:] # it's not data frame it's only capture the data
type(univ.data)
# Normalizing the numeric data
univ_normal = scale(univ.data)
univ_normal

pca = PCA(n_components = 6) # 6 columns - keep experiment if i pass more than 6 columns, less than 6
#  idealy we pass equal number of columns
pca_values = pca.fit_transform(univ_normal) # pca is done, transform value into pca
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
pca_data.columns = "pc0","pc1","pc2","pc03","pc04","pc05"

final = pd.concat([univ.Univ, pca_data.iloc[:, 0:3]], axis = 1) #univ.Univ 0th column

#Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.pc0, y = final.pc1)


