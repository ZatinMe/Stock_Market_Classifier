
# coding: utf-8

# In[1]:


import sys

print(sys.version)


# In[3]:


from pandas_datareader import data
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import numpy as np


# In[6]:


# The first task that needs to be done here is to load the companies

companies_dict = {
    'Amazon' : 'AMZN',
    'Apple' : 'AAPL',
    'Walgreen' : 'WBA',
    'Northrop Grumman' : 'NOC',
    'Boeing' : 'BA',
    'Lockheed Martin' : 'LMT',
    'McDonalds' : 'MCD' ,
    'Intel': 'INTC' ,
    'Navistar' : 'NAV',
    'IBM' : 'IBM' ,
    'Texas Instruments' : 'TXN',
    'MasterCard': 'MA',
    'Microsoft' : 'MSFT',
    'General Electric' : 'GE',
    'Symantec' : 'SYMC' , 
    'American Express' : 'AXP' ,
    'Pepsi' : 'PEP' , 
    'Coca Cola' : 'KO' ,
    'Johnson & Johnson' : 'JNJ' ,
    'Toyota' : 'TM', 
    'Honda' : 'HMC' ,
    'Mistubishi' : 'MSBHY' ,
    'Sony' : 'SNE' ,
    'Exxon' : 'XOM' ,
    'Chevron' : 'CVX' ,
    'Valero Energy' : 'VLO', 
    'Ford' : 'F' , 
    'bank of America' : 'BAC'
}

companies = sorted(companies_dict.items() , key = lambda x : x[1])


# In[23]:


# Using yahoo finance
import fix_yahoo_finance


# In[74]:



data_source = 'yahoo'

start_date = '2015-01-01'

end_date = '2017-12-31'

#Use pandas reader


# In[76]:



pandas_data = data.DataReader(list(companies_dict.values()), data_source, start_date,
                              end_date).unstack(level = 0)


# In[77]:


panel_data = pandas_data


# In[108]:


stock_close = panel_data.loc['Open']
stock_open = panel_data.loc['Close']


# 1
# 

# ## 

# 1
# 

# 2

# In[109]:


stock_close


# In[110]:


stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T


# In[114]:


row= stock_close.shape


# In[115]:


row


# In[117]:


stock_close


# In[118]:



stock_open


# In[119]:


movements = stock_close - stock_open


# In[120]:


movements


# In[123]:


plt.clf

plt.figure(figsize = (18 , 16))

ax1 = plt.subplot(221)
plt.plot(movements[0:767])
plt.title(companies[0])

plt.subplot(222, sharey = ax1)

plt.plot(movements[768:2 * 768 - 1])
plt.title(companies[1])


# In[157]:




movement = []
i = 0
condition = True
while(condition == True) :
    news = []
    
    for j in range(i , i + 756):
        news.append(movements[j])
        
    movement.append(news)
    i = i + 756
    if(i >= 21168) :
        break


# In[158]:


move = np.array(movement)


# In[159]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

new = normalizer.fit_transform(move)
print(new.max())
print(new.min())
print(new.mean())


# In[160]:


plt.clf

plt.figure(figsize = (18 , 16))

ax1 = plt.subplot(221)
plt.plot(move[0][:])
plt.title(companies[0])

plt.subplot(222, sharey = ax1)

plt.plot(move[1][:])
plt.title(companies[1])


# In[161]:


plt.clf

plt.figure(figsize = (18 , 16))

ax1 = plt.subplot(221)
plt.plot(new[0][:])
plt.title(companies[0])

plt.subplot(222, sharey = ax1)

plt.plot(new[1][:])
plt.title(companies[1])


# In[162]:


from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
normalizer = Normalizer()

# Kmeans model with 10 clusters

kmeans = KMeans(n_clusters = 10, max_iter = 1000)

pipeline = make_pipeline(normalizer , kmeans)


# In[163]:


pipeline.fit(move)

print(kmeans.inertia_)


# In[164]:


print(move.shape)


# In[165]:


print(len(companies))


# In[167]:


import pandas as pd

labels = pipeline.predict(move)

df = pd.DataFrame({'labels' : labels, 'Companies' : companies})

print(df.sort_values('labels'))

