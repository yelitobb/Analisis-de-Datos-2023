#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

diabetes


# In[4]:


df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df.head()


# In[5]:


df_target = pd.DataFrame(data=diabetes.target, columns=['target'])
df_target.head(3)


# In[6]:


# Definir nuestras variables dependiente e independiente(s)
X = df[['bmi']]
y = df_target

# Separar los datos en muestras de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[7]:


# Construir modelo de regresión lineal
modelo_regresion_simple = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_regresion_simple.fit(X_train, y_train)

# Predecir los valores de la variable objetivo (y) para los datos de prueba
y_pred = modelo_regresion_simple.predict(X_test)


# In[8]:


# Calcular el error cuadrático medio (MSE) de las predicciones
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Imprimir el MSE
print('Error cuadrático medio (MSE): {:.2f}'.format(mse))
print('r2: {:.2f}'.format(r2))


# In[9]:


print('pendiente:', modelo_regresion_simple.coef_)
print('intercepto:', modelo_regresion_simple.intercept_)


# In[10]:


m = modelo_regresion_simple.coef_
b = modelo_regresion_simple.intercept_
z = np.linspace(X.min(), X.max(), 100)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(X.values, y.values, 'ob', alpha=0.3)
ax.plot(z, m*z + b, ls='--', color='orange', lw=2)
plt.plot(X_test.values, y_pred, '^', color='orange', ms=8)
ax.set_xlabel('BMI')
ax.set_ylabel('disease progression');


# In[10]:


# Definir nuestras variables dependiente e independiente(s)
X2 = df[['bp']]
y2 = df_target

# Separar los datos en muestras de entrenamiento y de prueba
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# Construir modelo de regresión lineal
modelo_regresion_simple2 = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_regresion_simple2.fit(X2_train, y2_train)

# Predecir los valores de la variable objetivo (y) para los datos de prueba
y2_pred = modelo_regresion_simple.predict(X2_test)

m2 = modelo_regresion_simple2.coef_
b2 = modelo_regresion_simple2.intercept_
z2 = np.linspace(X2.min(), X2.max(), 100)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(X2.values, y2.values, 'ob', alpha=0.3)
ax.plot(z2, m*z2 + b2, ls='--', color='orange', lw=2)
plt.plot(X2_test.values, y2_pred, '^', color='orange', ms=8)
ax.set_xlabel('bp')
ax.set_ylabel('disease progression');


# In[49]:


from mpl_toolkits.mplot3d.axes3d import Axes3D

X3 = df[['bmi','bp']]

# Separar los datos en muestras de entrenamiento y de prueba
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.2, random_state=0)

# Construir modelo de regresión lineal
modelo_regresion_bivariable = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_regresion_bivariable.fit(X3_train, y3_train)

# Predecir los valores de la variable objetivo (y) para los datos de prueba
y3_pred = modelo_regresion_bivariable.predict(X3_test)


x_1 = np.linspace(X3['bmi'].min(),X3['bmi'].max(),100)
x_2 = np.linspace(X3['bp'].min(),X3['bp'].max(),100)
a = modelo_regresion_bivariable.coef_
plano = a[0][0]*x_1 + a[0][1]*x_2 + modelo_regresion_bivariable.intercept_
plano = np.array([plano]).reshape(100,1)
#plano = plano.reshape(100,1)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(X3['bmi'],X3['bp'], y, linewidth=0.2, color='blue', alpha=0.3)
ax.plot_surface(x_1,x_2,plano, color='black',alpha=0.2)
ax.scatter(X3_test['bmi'],X3_test['bp'], y3_pred, linewidth=0.2, color='red', alpha=0.3)
#ax.plot_surface(plano)


# In[48]:


X3_test['bmi']


# In[18]:


import seaborn as sns
sns.set_theme(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.jointplot(X['bmi'], y['target'],kind="reg", truncate=False,color="m", height=7)


# In[45]:




