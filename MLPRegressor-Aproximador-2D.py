#!/usr/bin/env python
# coding: utf-8

# # Paquetes a utilizar por la red Neuronal

# In[1]:



from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor 
from mpl_toolkits.mplot3d import axes3d
from sklearn import preprocessing
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np


# # Generacion de una sucesion de pares [ xi, yi, f(xi,yi) ]

# In[2]:



# Generamos los pares xi y yi (en este caso son dos variables de entrada (o input) a la red)

x = np.linspace(0.0, 4*np.pi, 150) #linspace para controlar que ambas entradas tengan el mismo numero de datos.
y = np.linspace(5.0, 6*np.pi-6.0, 150) 

xi = x[:,np.newaxis]
yi = y[:,np.newaxis]

#print(x,yi)


# # Seleccion de una funcion z = f(x,y)

# In[3]:



# La forma de la funcion sera la suma del argumento del sen(x + y)
z_sum = np.sin(xi+yi)


# ## Grafica de la funcion a aproximar 

# In[4]:


# Esta parte es una manera de auto-practicar como ver la data y los resultados graficamente

# Necesita el paquete mpl_toolkits.mplot3d.axes3d y matplotlib importcm para la superficie

X, Y = np.meshgrid(xi, yi) #Formato de coordenadas para graficar
Zsum = np.sin(X+Y) # Suma del argumento del sen(x + y)

fig = plt.figure(figsize=(16.5,7)) # Creacion de la figura 1 de tamaño 16.5x6

# Grafica 1: superficie de la funcion sen(x+y).

ax = fig.add_subplot(121,projection='3d')
ax.plot_surface(X, Y, Zsum, cmap=cm.viridis,linewidth=0, antialiased=False)
ax.set_title('Grafica 1: Superficie de la Funcion z = sen(x+y)', y=1)
ax.set_xlabel('X')
ax.set_xlim(0, 13)
ax.set_ylabel('Y')
ax.set_ylim(4, 13)
ax.set_zlabel('Z')
ax.set_zlim(-1.5, 1.5)

# Grafica 2: Proyeccion de la superficie z=sen(x+y) en los planos xy, xz, yz.
bx = fig.add_subplot(122,projection='3d')
cset = bx.contour(X, Y, Zsum, zdir='z', offset=-2, cmap=cm.viridis)
cset = bx.contour(X, Y, Zsum, zdir='x', offset=0, cmap=cm.viridis)
cset = bx.contour(X, Y, Zsum, zdir='y', offset=15, cmap=cm.viridis)
bx.set_title('Grafica 2: Proyecciones de la superficie z = sen(x+y) \n en los planos xy, xz, yz', y=1.0)
bx.set_xlabel('X')
bx.set_xlim(-1, 14)
bx.set_ylabel('Y')
bx.set_ylim(4, 14)
bx.set_zlabel('Z')
bx.set_zlim(-1.8,1.5)

plt.suptitle(' Graficas de la Funcion z = f(x,y) ',fontsize=16)
plt.show() # Ver figura


fig1 = plt.figure(figsize=(14,7)) # Creacion de la figura 2 de tamaño 14x5

# Grafica 3: Plano xz
plt.subplot(2, 2, 1)
plt.scatter(xi, z_sum, linewidth=1, color = 'blue')
plt.title('Grafica 3: Plano xz de la funcion z = sen(x+y) ')
plt.xlabel('x')
plt.ylabel('z')

#Grafica 4: Plano yz
plt.subplot(2, 2, 2)
plt.scatter(yi, z_sum, linewidth=1, color = 'green')
plt.title('Grafica 4: Plano yz de la funcion z = sen(x+y) ')
plt.xlabel('y')
plt.ylabel('z')

plt.show() # Ver figura


# # Division de la data en un conjunto de entrenamiento y un conjunto de validacion, usando una funcion build-in de Python.

# In[5]:


# Standarizamos la data para mejores resultados, necesita el paquete preprocessing de sklearn

datos = np.concatenate([xi, yi], axis = 1) # Formato para utilizar scikit-learn
data_scaled=preprocessing.StandardScaler().fit_transform(datos)

#print(datos,data_scaled)


# In[6]:


#Necesita el paquete sklearn.model_selection 

# Dividimmos la data, el porcentaje de datos para la prueba de la red es del 40%  para la funcion sen(x+y)
X_train_sum, X_test_sum, y_train_sum, y_test_sum = train_test_split(data_scaled, z_sum, test_size=0.4)


# # Seleccion y entrenamiento de la red neural con una capa escondida, usando una funcion built-in de Python.

# In[7]:


# Necesita el paquete sklearn.neural_network

# Seleccionamos una red de retropropagación, un Perceptron Multi-capa para un modelo de regresion no lineal.
regr = MLPRegressor(hidden_layer_sizes=(100,50),max_iter = 5000, activation='tanh',solver = 'lbfgs')

# Usamos la funcion de activacion tanh y el numero de capas ocultas de 
#la red es 2, las cuales poseen 100 y 50 neuronas, respectivamente.

# Entrenamos la red con la funcion sen(x+y)
regr.fit(X_train_sum, y_train_sum.ravel())


# # Evaluacion del desempeño de la red

# In[8]:



#Realizamos las predicciones 
Z_prediciones_sum = regr.predict(X_test_sum)

#Observamos el error (En este caso se calcula como R-cuadrado (r**2), es decir, mientras mas cercano a 1 mejores resultados)
Z_error_sum = regr.score(X_test_sum, y_test_sum)

print('Sen(x+y) Error: ' + str(Z_error_sum))


# ## Grafica de las predicciones vs los datos de entrenamiento 

# In[9]:


# Manipulamos la data para obtener el formato deseado
xi_train = X_train_sum[:,0]
yi_train = X_train_sum[:,1]
zi_train = y_train_sum

xi_test = X_test_sum[:,0]
yi_test = X_test_sum[:,1]
zi_test = Z_prediciones_sum

#Graficamos las prediciones en conjunto a los datos de entrenamiento.
fig2 = plt.figure(figsize=(15,8)) # Creacion de la figura 3 de tamaño 15x5

plt.suptitle((' Datos de Entrenamiento vs Predicciones de la Red \n El error de la red fue: ' + str(Z_error_sum)),fontsize=15)

# Grafica 5: Plano xz
plt.subplot(1, 2, 1)
plt.scatter(xi_train, zi_train, linewidth=1, color = 'blue',label='Data de Entrenamiento')
plt.scatter(xi_test, zi_test, color = 'red',s=30, linewidth=0.01, label='Predicciones de la Red')
plt.title('Grafica 5: Plano xz de la funcion z = sen(x+y) ')
plt.xlabel('x')
plt.ylabel('z')
plt.axis([-1.5, 1.5, -1.25, 1.5])
plt.legend(loc="upper right")

#Grafica 6: Plano yz
plt.subplot(1, 2, 2)
plt.scatter(yi_train, zi_train, linewidth=1, color = 'green', label='Data de Entrenamiento')
plt.scatter(yi_test, zi_test, color = 'red',s=30, linewidth=0.1, label='Predicciones de la Red')
plt.title('Grafica 6: Plano xy de la funcion z = sen(x+y) ')
plt.xlabel('y')
plt.ylabel('z')
plt.axis([-1.5, 1.5, -1.25, 1.5])
plt.legend(loc="upper right")

plt.show()

