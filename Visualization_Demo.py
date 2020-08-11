#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# # Visualizing Time Series in Pandas Demo

# In[2]:


get_ipython().system('wget --no-verbose -nc https://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/CalIt2.data;')
get_ipython().system('wget --no-verbose -nc https://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/CalIt2.events;')
get_ipython().system('wget --no-verbose -nc https://archive.ics.uci.edu/ml/machine-learning-databases/event-detection/CalIt2.names;')


# ## Dataset
# 
# https://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts
# 

# **Raw Data**

# In[3]:


pd.read_csv('CalIt2.data', header=None, names=['Flow', 'Date', 'Time', 'Count'])


# ### Process Data

# In[4]:


def load_data(filepath):
    df = pd.read_csv('CalIt2.data', header=None, names=['Flow', 'Date', 'Time', 'Count'])
    
    # Process times
    df['Timestamp'] = df['Date'] + ' ' + df['Time']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    df['Date'] = df['Timestamp'].dt.date
    df['Time'] = df['Timestamp'].dt.time
    
    
    # Process Flow Column
    df.loc[df['Flow']==7, 'Flow'] = 'Out'
    df.loc[df['Flow']==9, 'Flow'] = 'In'
    
    df = df.set_index('Timestamp')
    return df
    

df = load_data('CalIt2.data')


# In[5]:


df.head()


# ### Pivot to get separate columns for Inflow and Outflow

# In[6]:


def pivot_data(df):
    
    df_pivot = df.pivot_table(index='Timestamp', columns=['Flow'])
    df_pivot.columns = ['In', 'Out']
    
    df_pivot['Net'] = df_pivot['In'] - df_pivot['Out']
    
    df_pivot = df_pivot.reset_index()
    
    df_pivot['Date'] = df_pivot['Timestamp'].dt.date
    df_pivot['Time'] = df_pivot['Timestamp'].dt.time
    
    df_pivot = df_pivot.set_index('Timestamp')
    
    return df_pivot


# In[7]:


df_pivot = pivot_data(df)
df_pivot.head()


# ## Cumulative flow throughout day

# In[8]:



def daily_cumulative(df_pivot):

    return (df_pivot
      .groupby('Date')
      .cumsum())

(daily_cumulative(df_pivot)).head()


# In[9]:


(daily_cumulative(df_pivot)
  .loc['2005-07-28', 'In']
  .plot())


# ## Resample

# In[10]:


def resample_pivot(df_pivot, sampling='h'):
    
    return (df_pivot
            .resample(sampling)
            .agg({'In':'sum',
                  'Out': 'sum',
                  'Net':'sum',
                  'Date': 'last',
                  'Time': 'first'}))


df_hourly = resample_pivot(df_pivot)


# In[11]:


def hourly_with_confidence_bars():
    df_hourly = resample_pivot(df_pivot)
    df_g = (df_hourly
     .groupby('Time')
     .agg({'In':['mean', 'std']}))

    df_g.columns = df_g.columns.get_level_values(1)


    ax = df_g.plot(y='mean')

    ax.fill_between(df_g.index, 
                    df_g['mean']+df_g['std'],
                    df_g['mean']-df_g['std'],
                    alpha=0.3)

hourly_with_confidence_bars()


# ## By month

# In[12]:


def resample_pivot_month(df_pivot, sampling='m'):
    return (df_pivot
            .resample(sampling)
            .agg({'In':'mean',
                  'Out': 'mean',
                  'Net':'mean',
                  'Date': 'last',
                  'Time': 'first'}))


# In[13]:


def get_monthly(df_hourly):
    
    monthly = df_hourly.reset_index()
    
    monthly['Month'] = monthly['Timestamp'].dt.month
    
    return monthly.set_index('Timestamp')


# In[14]:



monthly = get_monthly(df_hourly).reset_index()
monthly['Month'] = monthly['Timestamp'].dt.month

monthly = monthly.set_index('Timestamp')


# In[15]:


monthly


# In[16]:


def get_monthly_inflow(monthly):
    
    monthly_in = monthly.loc[:,'In']
    
    monthly_in_groups = monthly_in.reset_index().melt(id_vars='Timestamp').groupby('Timestamp').groups
    monthly_in_melt = monthly_in.reset_index().melt(id_vars='Timestamp')
    monthly_in_melt['Month'] = monthly_in_melt['Timestamp'].dt.month    
    
    
    return monthly_in_melt


# In[17]:


def month_sparkline(df, max_month='November'):
    
    months = {
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November'}
    
    
    fig, ax = plt.subplots(1,1,figsize=(4,1))
    df.plot(x='Time', y='value', ax=ax, legend=None)

    month = months[df['Month'][0]]


    # remove all the axes
    for k,v in ax.spines.items():
        v.set_visible(False)
    if month is not 'November':
        plt.xlabel('')
    
    
        ax.set_xticks([])
    ax.set_yticks([])
    #plt.axis('off')
    
    plt.ylabel(month, rotation='horizontal', labelpad=40, size=10)
    # remove legend
    
    


# In[ ]:





# In[ ]:




