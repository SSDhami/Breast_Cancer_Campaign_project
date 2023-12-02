import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv('data.csv')#Read the dataset into a Pandas DataFrame!





#Choose the features you think are relevant to our analysis! There are a lot of features in this dataset but we have to make our model’s training time reasonable for you.
#Hint: Notice the fact that some of the data in the Breast Cancer dataset is irrelevant to the research such as the id attribute.
df = df.drop(['id'], axis=1)




#Preprocessing
#Perform any needed pre-processing on the chosen features including:
#Scaling


column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst',
                'symmetry_worst', 'fractal_dimension_worst']

df[column_names]=StandardScaler().fit_transform(df[column_names].values)


#Encoding
df["diagnosis"]=LabelEncoder().fit_transform(df["diagnosis"].values)


#Dealing with Nan values

df = df.dropna(axis =1)


#Note:
#You need to output the result of your pre-processing to an output CSV called “data_refined.csv”.
df.to_csv('data_refined.csv')  


#Visualization
#You need to deliver a number of visualizations for your dataset including:

#Pair Plots for the features.

sns.pairplot(data=df, x_vars=df.columns)
plt.show()

#Correlation Matrix heat map.
plt.figure(figsize=(35,15))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()
sns.heatmap(df.corr())

#Box plots for the features.

plt.figure(figsize=(35,30))
plt.xticks(rotation=70,fontsize=15)
plt.tight_layout()
sns.boxplot(data=df)

#Visualize your data in violin plots.
plt.figure(figsize=(35,15))
plt.xticks(rotation=70,fontsize=30)
plt.tight_layout()
sns.violinplot(data = df,width =2) 

df



