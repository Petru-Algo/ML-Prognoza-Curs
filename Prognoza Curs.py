#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd


# In[2]:


df = pd.read_excel("EURO.xls")


# In[7]:


df.describe()


# In[6]:


df ["curs"].mean()


# In[13]:


df ["curs_ban"] = df ["curs"] * 100


# In[14]:


df 


# In[15]:


df["curs"].plot()


# In[27]:


money = df["curs"]


# In[28]:


money.plot()


# In[18]:


past = 7 * 4 # interval in trecut
future = 7 # interval prognozat 


# In[29]:


len(money)


# In[30]:


start = past 
end = len (money) - future


# In[33]:


raw_df = []
for i in range(start,end):
    past_and_future_values = money[(i-past):(i+future)]
    raw_df.append(list(past_and_future_values))


# In[38]:


past_columns = []
for i in range (past):
    past_columns.append("past_{}".format(i))


# In[41]:


future_columns = []
for i in range (future):
    future_columns.append("future_{}".format(i))


# In[45]:


df = pd.DataFrame(raw_df, columns=(past_columns+future_columns))


# In[46]:


df 


# In[50]:


# Invatare

x = df[past_columns] [:-20]
y = df[future_columns] [:-20]


# In[52]:


# Verificare

x_test = df[past_columns] [-20:]
y_test = df[future_columns] [-20:]


# In[54]:


from sklearn.linear_model import LinearRegression


# In[55]:


LinReg = LinearRegression()


# In[56]:


LinReg.fit(x, y)


# In[57]:


prediction = LinReg.predict(x_test)


# In[58]:


prediction


# In[60]:


prediction[19]


# In[61]:


y_test [-1:]


# In[62]:


import matplotlib.pyplot as plt


# In[87]:


plt.plot(prediction[2], label="prediction")
plt.plot(y_test[-18:].iloc[0], label="real")
plt.legend()


# In[69]:


from sklearn.metrics import mean_absolute_error


# In[88]:


mean_absolute_error(y_test[-18:].iloc[0], prediction[5])


# In[92]:


def printErrors(prediction):
    errors_list = []
    for i in range (len (prediction)):
        error = mean_absolute_error (y_test.iloc[i], prediction[i])
        errors_list.append(error)
    avg_err = sum(errors_list) / len (errors_list)
    max_err = max(errors_list)
    print("eroarea medie = {}".format(avg_err))
    print("eroarea maxima = {}".format(max_err))


# In[93]:


printErrors(prediction)


# In[94]:


from sklearn.neural_network import MLPRegressor


# In[95]:


MLP = MLPRegressor()


# In[96]:


MLP.fit(x,y)


# In[99]:


predictionMLP = MLP.predict(x_test)


# In[109]:


printErrors(predictionMLP)


# In[124]:


from sklearn.neighbors import KNeighborsRegressor


# In[139]:


KNN = KNeighborsRegressor(n_neighbors=45)


# In[140]:


KNN.fit(x,y)


# In[141]:


predictionKNN = KNN.predict(x_test)


# In[142]:


printErrors(predictionKNN)


# In[143]:


plt.plot(predictionKNN[19], label="predictionKNN")
plt.plot(y_test[-1:].iloc[0], label="real")
plt.legend()


# #### 

# In[29]:


from sklearn.model_selection import GridSearchCV 


# In[30]:


from sklearn.neural_network import MLPRegressor 


# In[31]:


MLP = MLPRegressor(hidden_layer_sizes = (100,100,100))  


# In[32]:


MLP.get_params().keys() 


# In[28]:


MLP = MLPRegressor()
GSCV = GridSearchCV (MLP, {
    "max_iter": [100,500,1000],
    "learning_rate_init": [0.001, 0.01],
    "hidden_layer_sizes" : [(100,100,100), (50,50), (10)]
}, cv=3, scoring = 'neg_mean_absolute_error')


# In[26]:


GSCV.fit(X,y)


# In[ ]:


GSCV.best_estimator_

