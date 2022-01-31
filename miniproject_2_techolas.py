#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# In[2]:


data=pd.read_excel("Nanomaterials.xlsx")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[20]:


data=data.drop(["Index","Ligand1 SMILES","Ligand2 SMILES","Ligand3 SMILES","Ligand4 SMILES"],axis=1)


# In[21]:


label_encoder=preprocessing.LabelEncoder()


# In[22]:


data['Core']=label_encoder.fit_transform(data['Core'])


# In[23]:


data['Shape']=label_encoder.fit_transform(data['Shape'])


# In[24]:


data['Size'] = label_encoder.fit_transform(data['Size'].astype(str))


# In[25]:


data['Size']=label_encoder.fit_transform(data['Size'])


# In[26]:


data


# In[27]:


mean1=data["Zeta potential in water (mv)"].mean()


# In[28]:


mean1


# In[29]:


data["Zeta potential in water (mv)"]=data["Zeta potential in water (mv)"].fillna(mean1)


# In[30]:


mean2=data["Cellular uptake in A549 (106 nm2 cell-1)"].mean()


# In[31]:


mean2


# In[32]:


data["Cellular uptake in A549 (106 nm2 cell-1)"]=data["Cellular uptake in A549 (106 nm2 cell-1)"].fillna(mean2)


# In[33]:


mean3=data["logP"].mean()


# In[34]:


mean3


# In[35]:


data["logP"]=data["logP"].fillna(mean3)


# In[36]:


data


# In[37]:


data["#Ligand1"]=data["#Ligand1"].replace("-",np.nan)
data["#Ligand2"]=data["#Ligand2"].replace("-",np.nan)
data["#Ligand3"]=data["#Ligand3"].replace("-",np.nan)
data["#Ligand4"]=data["#Ligand4"].replace("-",np.nan)


# In[38]:


data


# In[39]:


data=data.fillna(0)
data


# In[40]:


d=normalize(data)


# In[41]:


df_new=pd.DataFrame(d,columns=data.columns)


# In[42]:


df_new


# In[43]:


corr=data.corr()


# In[44]:


plt.figure(figsize=(20,15))
sns.heatmap(data=corr,annot=True)


# In[45]:


x=df_new.drop(["logP","Zeta potential in water (mv)","Cellular uptake in A549 (106 nm2 cell-1)",],axis=1)


# In[46]:


y=df_new["logP"]


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)


# In[48]:


model=KNeighborsRegressor(7)


# In[49]:


model.fit(x_train,y_train)


# In[50]:


model.score(x_test,y_test)


# In[51]:


leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn = KNeighborsRegressor()
#Use GridSearch
gscv = GridSearchCV(knn, hyperparameters, cv=10)
#Fit the model
best_model = gscv.fit(x,y)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[52]:


best_model.score(x_test,y_test)


# In[53]:


y2=df_new["Zeta potential in water (mv)"]


# In[54]:


x_train,x_test,y2_train,y2_test=train_test_split(x,y2,test_size=0.2,random_state=9)


# In[55]:


model2=KNeighborsRegressor(1)


# In[56]:


model2.fit(x_train,y2_train)


# In[57]:


model2.score(x_test,y2_test)


# In[58]:


leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn2 = KNeighborsRegressor()
#Use GridSearch
gscv2 = GridSearchCV(knn2, hyperparameters, cv=10)
#Fit the model
best_model2= gscv2.fit(x,y2)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[59]:


best_model2.score(x_test,y2_test)


# In[60]:


y3=df_new["Cellular uptake in A549 (106 nm2 cell-1)"]


# In[61]:


x_train,x_test,y3_train,y3_test=train_test_split(x,y3,test_size=0.2,random_state=9)


# In[62]:


model3=KNeighborsRegressor()


# In[63]:


model3.fit(x_train,y3_train)


# In[64]:


model3.score(x_test,y3_test)


# In[65]:


leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn3 = KNeighborsRegressor()
#Use GridSearch
gscv3 = GridSearchCV(knn3, hyperparameters, cv=10)
#Fit the model
best_model3= gscv3.fit(x,y3)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[66]:


best_model3.score(x_test,y3_test)


# In[ ]:





# In[68]:


import pickle


# In[70]:


#save the model as a pickle in the file
pickle.dump(best_model3,open('saved_model.ipynb','wb'))


# In[ ]:





# In[ ]:




