#!/usr/bin/env python
# coding: utf-8

# I downloaded customers dataset from UCI Machine Learning Repository. with focus on the six product categories recorded for customers

# In[23]:


# Import libraries necessary for this project
import numpy as np
import pandas as pda
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import plotting as pt

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# ## Data Exploration
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', and 'Delicatessen'. Consider what each category represents in terms of products you could purchase.

# In[3]:


display(data.describe())


# In[4]:


indices = [26,176,392]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# ### Implementation: Feature Relevance
# 
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.

# In[5]:


new_data = data.drop('Fresh',axis=1)
label=data['Fresh']
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(new_data,label,test_size=0.25,random_state=42)


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


score = r2_score(y_test,y_pred)
print('Regressor has an prediction score of {}'.format(score))


# I attempted to predict 'Fresh' feature.
# 
# The prediction score of the regressor is -0.33,
# 
# Based on the R2 score we can say that the model has failed to fit the data. Which means that rest of the features doesnt help in predicting the feature. Since this feature cannot be predicted, we can say that this feature is necessary to identify customer spend habits.

# In[7]:


pip install seaborn


# ## Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data.

# In[6]:


import seaborn as sns
sns.pairplot(data,diag_kind = 'kde')


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True,fmt='.2f')
plt.show()


# From the scatter matrix and the heatmap it is evident that majority of the data doesnt follow a bivariate normal distrbution. Few features seem to show a skewed normal distrbution along its own marginal axes which means that a particular feature isnt effected by the other. Take detergent_paper and delicatessen for example. Data seems to be spread only along detergent_paper indicating that spend on delicatessen doesnt relate to spend on detergent_paper on the other hand, few features show a distribution along both marginal axes which means that those features are contrast to each other. Take detergent_paper and Fresh for instance, from the plot we can infer that people who spend more on Fresh might not spend much on detergent_paper and vice versa. Hence we can say that the features doesnt follow a Normal distribution. we can also see that the data is more skewed towards the center. This can be due to the dataset not being scaled and these outliers can impact any model.
# 
# The pair of features that seem to exhibit some correlation are below -
# 
# Grocery and deteregents_paper - compared to all others this pair to seem to show high degree of correlation. It is normally distributed but the data is skewed at the origin
# 
# Milk - Grocery and Milk - detergent_paper exhibit mild correlation as we can see. It shows variance on both features but still follows a normal distribution.
# 
# This confirms my statement about the relevance of the feature I attempted to predict. we can see that the feature i attempted to predict i.e. Fresh doesnt show any kind of relevance to other features. So it is hard to predict feature 'Fresh' given other features.

# ## Data Preprocessing
# 
# ### Implementation: Feature Scaling

# In[26]:


log_data = np.log(data)


log_samples = np.log(samples)


# In[9]:


sns.pairplot(log_data,diag_kind = 'kde')


# In[10]:


# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# 
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. An outlier step is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# Assign the value of the 25th percentile for the given feature to Q1. Use np.percentile for this.
# Assign the value of the 75th percentile for the given feature to Q3. Again, use np.percentile.
# Assign the calculation of an outlier step for the given feature to step.
# Optionally remove data points from the dataset by adding indices to the outliers list.

# In[24]:


for feature in log_data.keys():
    
    
    Q1 = np.percentile(log_data[feature],25)
    print(Q1)
    
    
    Q3 = np.percentile(log_data[feature],75)
    print(Q3)
    
    
    step = 1.5 * (Q3-Q1)
    print(step)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [66, 75, 338, 142, 154, 289]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# In[12]:


sns.pairplot(good_data,diag_kind = 'kde')


# ## Feature Transformation
# 
# ### Implementation of PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the good_data to discover which dimensions about the data best maximize the variance of features involved.

# In[13]:


from sklearn.decomposition import PCA
pca = PCA()

pca.fit(good_data)


# In[14]:


pca.n_components_


# In[15]:


#Plotting the Cumulative Summation of the Explained Variance
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()


# In[25]:


pca_samples = pca.transform(log_samples)

pca_results = pt.pca_results(good_data, pca)


# In[17]:


print(pca.explained_variance_ratio_)


# In[18]:


print(pca.components_)


# Variance of 1st PC - 0.44302505
# Variance of 2nd PC - 0.26379218 Total Variance of 1st two PCs = 0.44302505 + 0.26379218 = ~70.6%
# 
# Variance of 3rd PC - 0.1230638 Variance of 4th PC - 0.10120908 Total Variance of 1st four PCs = 0.44302505 + 0.26379218 + 0.1230638 + 0.10120908 = 93.1%
# 
# I'd guess the negative weights indicate the direction of variance of that feature in that dimension. That being said,
# 
# The first PC has shown and variance of 0.44. so, looking at the values of the vectors, this dimension has captured good variance of Milk, grocery and detergent_paper. This behavior is same as one of the samples I picked in question #1. This dimension could represent a supplier, retailer etc. It also shows that the other two features(Fresh and frozen) are in perfect contrast with the other features, the means customer may not spend on those as per first dimension
# 
# The second PC has captured good variation of Fresh, Frozen and delicatessen. It also captured decent variation in Milk feature along the direction as other features which means all these features are relevant to each other. This dimension could represent a restaurant because it needs all features except detergent_paper to prepare food and restaurant also uses detergent_paper for cleaning etc which also is shown as relevant to other features.
# 
# The Third PC captured variance on delicatessen, Fresh and frozen with fresh being completely contrast to other two features. This could represent a take away shop like in the aiports which has delicacies, fresh salads and frozen foods ready to eat.
# 
# Fourth PC captures variance of frozen foods and delicatessen with both showing complete contrast in relevance. so customer might be spending on frozen foods or delicatessen. so it can represnt a small store

# In[19]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# 
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem.

# In[20]:


pca = PCA(n_components=2)
pca.fit(good_data)

reduced_data = pca.transform(good_data)

pca_samples = pca.transform(log_samples)


reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# In[21]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# In[22]:


# Create a biplot
pt.biplot(good_data, reduced_data, pca)

