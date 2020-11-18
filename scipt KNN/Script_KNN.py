import numpy as np
import pandas as pd
import statistics as st
import math
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21 #consumo es una lista
    mtcars.mpg[ix_consumo_alto] = 1
    mtcars.mpg[~ix_consumo_alto] = 0
    #en la columna mpg ya tengo si es 0 o es 1
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1)
    # Añadimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 
    
    train, test = splitTrainTest(mtcars_normalizado, 0.75)

    # Separamos las labels del Test. Es como si no nos las dieran!!

    lista_test = []

    for i in range(0, len(test)):
        lista_test.append(list(test.iloc[i]))

    #print(lista_test)

    # Predecimos el conjunto de test
    true = []
    pred = []
    for i in range(0, len(test)):
        clase = knn(test.iloc[i], train, 3)
        print("La clase es: "+str(clase))
        print("La clase real es: "+str(test.iloc[i].mpg))

    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    #print(accuracy(true_labels, predicted_labels))

    # Algun grafico? Libreria matplotlib.pyplot
    return(0)

# FUNCIONES de preprocesado
def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x):
    return((x-st.mean(x))/st.variance(x))

# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    """
    Takes a pandas dataframe and a percentaje (0-1)
    Returns both train and test sets
    """
    #si el 1 es la longitud completa, 0.75 será 3/4 de la longitud
    longitud = len(data)
    percentaje = int(longitud*percentajeTrain)
    training_set = data[0:percentaje]
    test_set = data[percentaje:longitud]

    return(training_set, test_set)

def kFoldCV(data, K):
    """
    Takes a pandas dataframe and the number of folds of the CV
    YOU CAN USE THE sklearn KFold function here
    How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    return()

# FUNCION modelo prediccion
def knn(newx, data, K):
    """
    Receives two pandas dataframes. Newx consists on a single row df.
    Returns the prediction for newx
    """

    #newx es una fila de la tabla y data es la tabla (matriz)

    euclideanList = []
    lista_distancias_ordenada = [] #lista que contiene distancias junto con sus casos
    lista_vecinos = []
    cont_1 = 0
    cont_0 = 0

    list_newx = list(newx)

    for i in range(0, len(data)): 
        list_data = list(data.iloc[i])
        euclideanList.append([euclideanDistance2points(list_newx, list_data), data.mpg[i]]) 
    
    #ordenamos la lista de menor a mayor distancias
    lista_distancias_ordenada = sorted(euclideanList)
    lista_vecinos = lista_distancias_ordenada[0:K]

    #contar cuantas clases tiene cada caso
    #si es mayor que 21, su clase es 1
    for j in range(0, len(lista_vecinos)):
        if lista_vecinos[j][1] == 1:
            cont_1 += 1
        else:
            cont_0 += 1
        
    #asignar clase al nuevo caso

    return 1 if cont_1>cont_0 else 0

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    x, y = np.array(x), np.array(y)
    #np.linalg.norm -> distancia euclidea entre 2 vectores
    return np.linalg.norm(x-y)

# FUNCION accuracy
def accuracy(true, pred):

    return (true[0]+true[1])/(pred[0]+pred[1]+pred[2]+pred[3])

if __name__ == '__main__':
    np.random.seed(25)
    main()


# path_dataset = "mtcars.csv" # Escoged bien la ruta!!
# data = pd.read_csv(path_dataset) # Leemos el csv
# # Discretizamos la variable clase para convertirlo en un problema de clasificacion
# ix_consumo_alto = data.mpg >= 21 #consumo es una lista
# data.mpg[ix_consumo_alto] = 1
# data.mpg[~ix_consumo_alto] = 0
# train, test = splitTrainTest(data, 0.75)
# lista_test = list(test.iloc[0])
# print(list(test.iloc[0]))
# # print(lista_test)
# # print(train)
# # print(len(test))
# # for i in range(0, 10):
# #     print(list(data.iloc[i]))
#me falta dividir el data set en train y test de manera aleatoria


