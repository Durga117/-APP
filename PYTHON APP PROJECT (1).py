#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import warnings
warnings.filterwarnings('ignore')


# # 1. Import required libraries and read the dataset.

# In[2]:


da = pd.read_csv("Apps_data+(1) (1).csv")
da


# # 2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features.

# In[3]:


da.head(10)


# In[4]:


da.shape


# In[5]:


da.size


# In[6]:


da.info()


# In[7]:


da.isnull().sum()


# # 3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model building.

# In[8]:


summary_stat = da.describe().T

listing_columns = []

print(summary_stat)
print(listing_columns)


# # 4. . Check if there are any duplicate records in the dataset? if any drop them.

# In[9]:


x = da.duplicated()
sum(x)


# In[10]:


x = da.drop_duplicates()
x


# # 5. Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them.

# In[11]:


x['Category'].unique()


# In[12]:


x['Category'].value_counts()


# In[13]:


y = x['Category'] != 1.9
x= x[y]


# In[14]:


invalid_mask = x.isnull().any(axis=1)

# Apply the mask to remove rows with invalid data
x = x[~invalid_mask]


# In[15]:


x['Category'].value_counts()


# # 6. Check if there are missing values present in the column Rating, If any? drop them and and create a new
# column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)

# In[16]:


da = pd.read_csv("Apps_data+(1) (1).csv")
da

missing_values = x['Rating'].isnull().sum()
print("Count of Missing_values:", missing_values)

x['Rating'] = da['Rating']


# In[17]:


x = x.dropna(subset=['Rating'])


# In[18]:


x['Rating_category'] = x['Rating'].apply(lambda x: 'High' if x > 3.5 else 'Low')

x['Rating_category']


# In[19]:


x


# # 7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution.

# In[20]:


x['Rating_category'].value_counts()

In the Rating_category we can see that there is High ratings. It has 8007 High ratings, and 879 Low rating 
# # 8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and handle the outliers using a transformation approach.(Hint: Use log transformation)

# In[21]:


x['Reviews'].value_counts()


# In[22]:


x['Reviews'] = pd.to_numeric(x['Reviews'], errors='coerce')
x['Reviews']


# In[23]:


plt.figure(figsize=(4, 3))
sns.boxplot(x=x['Reviews'])
plt.title("Box Plot of Reviews")
plt.show()


# In[24]:


Q1 = x['Reviews'].quantile(0.25)
Q3 = x['Reviews'].quantile(0.75)
IQR = Q3 - Q1

IQR_threshold = 1.5

outliers_low = x['Reviews'] < (Q1 - IQR_threshold * IQR)
outliers_high = x['Reviews'] > (Q3 + IQR_threshold * IQR)


x.loc[outliers_low, 'Reviews'] = np.log1p(x.loc[outliers_low, 'Reviews'])
x.loc[outliers_high, 'Reviews'] = np.log1p(x.loc[outliers_high, 'Reviews'])

outliers = x[outliers_low | outliers_high]

if outliers.empty:
    print('No outliers found')
else:
    print("Outliers found:",len(outliers))


# In[25]:


x['Reviews'] = np.log1p(x['Reviews']) 
x['Reviews'].head(20)


# # 9.. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into suitable data type. (hint: Replace M with 1 million and K with 1 thousand, and drop the entries where size='Varies with device')

# In[26]:


x['Size'] = x['Size'].replace({'M':'1e6','K':'1e3','Varies with device':'NaN'},regex= True)
x['Size'] = pd.to_numeric(x['Size'],errors='coerce')

x= x.dropna(subset = ['Size'])
x['Size'].head()


# # 10. Check the column 'Installs', treat the unwanted characters and convert the column into a suitable data type.

# In[27]:


x['Installs'].unique()


# In[28]:


x['Installs'] = x['Installs'].str.replace('[+,]','', regex = True)
x['Installs'].head()

x['Installs'].astype(int)


# # 11. . Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type.

# In[29]:


x['Price'].unique()


# In[30]:


x['Price'].astype(str)
x['Price'] = x['Price'].str.replace('[$]','',regex=True)
x['Price'].head()
x['Price'].astype(float)


# # 12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated', 'Current Ver','Android Ver' columns since which are redundant for our analysis)
# 

# In[31]:


Droping_columns = ['App','Rating','Genres','Last Updated','Current Ver','Android Ver']
x = x.drop(columns = Droping_columns)


# In[32]:


x.head()


# # 13. Encode the categorical columns.

# In[33]:


x['Content Rating'].value_counts()


# In[34]:


x['Type'].value_counts()


# In[35]:


x = pd.get_dummies(x,columns=['Category'],prefix = 'Category')


# In[36]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x['Type'] = label_encoder.fit_transform(x['Type'])
x['Content Rating'] = label_encoder.fit_transform(x['Content Rating'])
x['Rating_category'] = label_encoder.fit_transform(x['Rating_category'])


# In[37]:


x


# # 14. Segregate the target and independent features (Hint: Use Rating_category as the target)

# In[38]:


from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[39]:


X = x.drop("Rating_category", axis=1)  
y = x["Rating_category"]  


# # 15. Split the dataset into train and test.

# In[40]:


X = x.drop("Rating_category", axis=1)  
y = x["Rating_category"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)


# # 16. Standardize the data, so that the values are within a particular range.

# In[41]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to your training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scale

X_test_scaled = scaler.transform(X_test)

