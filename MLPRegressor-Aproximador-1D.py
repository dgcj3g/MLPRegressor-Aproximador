#!/usr/bin/env python
# coding: utf-8

# # Paquetes a utilizar por la red Neuronal

# In[1]:



from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor 
import matplotlib.pyplot as plt
import numpy as np 


# # Seleccion de una funcion y = f(x) y generacion de una sucesion de pares      (xi, f(xi)).

# In[2]:



# Rango de la funcion
x = np.arange(0.0, 6*np.pi, 0.2) # usamos Pi debido la periodicidad de la funcion

xi=x[:,np.newaxis] # Este formato es necesario para utilizar la data que se genero en la variable x en scikit-learn

# Forma de la funcion
y = np.sin(xi) 


# ## Grafica de la funcion a aproximar 

# In[3]:



#plt.plot(xi,y, color = 'red', linewidth=0.9)

plt.scatter(xi,y,linewidth=0.2, color = 'violet')

plt.title('Función f(x) = senx')
plt.xlabel('xi')
plt.ylabel('f(Xi)')

plt.show()


# # Division de la data en un conjunto de entrenamiento y un conjunto de validacion, usando una funcion build-in de Python.

# In[4]:


#Necesita el paquete sklearn.model_selection 

# Dividimmos la data, usamos test_size para definir el porcentaje de datos para entrenamiento, en este caso 40%
X_train, X_test, y_train, y_test = train_test_split(xi, y, test_size=0.4)


# # Seleccion y entrenamiento de la red neural con una capa escondida, usando una funcion built-in de Python.

# In[5]:


# Necesita el paquete sklearn.neural_network

# Seleccionamos una red de retropropagación, un Perceptron Multi-capa para un modelo de regresion no lineal.
regr = MLPRegressor(hidden_layer_sizes=(100,),max_iter = 5000, activation='tanh',solver = 'lbfgs')

#Usamos la funcion de activacion tanh y el numero de capas ocultas de la red es 1, la cual posee 100 neuronas.

# Entrenamos la red
regr.fit(X_train, y_train.ravel())

# Utilizamos ravel() para darle un formato valido a la variable y en scikit-learn


# # Evaluacion del desempeño de la red

# In[6]:



#Realizamos las predicciones
Y_prediciones = regr.predict(X_test)

#Observamos el error (En este caso se calcula como R-cuadrado (r**2), es decir, mientras mas cercano a 1 mejores resultados)
regr.score(X_test, y_test)

print('Error: ' + str(regr.score(X_test, y_test)))


# ## Grafica de las predicciones vs los datos de entrenamiento 

# In[7]:


#Graficamos las prediciones en conjunto a los datos de entrenamiento
plt.scatter(X_train, y_train, linewidth=0.2, color = 'violet')
plt.scatter(X_test, Y_prediciones, color='blue', linewidth=3)
plt.title('Regresión no Lineal Simple de y = sinx con error=' + "{0:.3f}".format(regr.score(X_test, y_test)))
plt.xlabel('Xi')
plt.ylabel('f(Xi)')
plt.show()

