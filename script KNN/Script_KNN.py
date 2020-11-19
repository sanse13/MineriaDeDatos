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
    # randomized = mtcars.sample(len(mtcars), replace=False)
    # print(randomized)
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
    
    randomized = mtcars_normalizado.sample(len(mtcars_normalizado), replace=False)
    
    train, test = splitTrainTest(randomized, 0.75)

    # print("El train es:")
    # print(train)
    # print("\n")
    # print("El test es:")
    # print(test)
    # print("\n\n")

    # Separamos las labels del Test. Es como si no nos las dieran!!
    for i in range(0, len(test)):
        print("Label "+str(i)+": \n"+str(test.iloc[i]))
        print("\n\n")

    # Predecimos el conjunto de test
    true, pred = 0, 0

    for i in range(0, len(test)):
        punto_predicho = knn(test.iloc[i], train, 3)
        print("El punto predicho del Label "+str(i)+" es: ")
        print(punto_predicho)
        print("\n")
        # print("El caso real es: "+str(test.mpg[i]))
        # print("La clase predicha con knn es: "+str(clase))
        if punto_predicho.mpgPred == punto_predicho.mpg:
            true += 1
        pred += 1

    print("Los trues son "+str(true))
    print("Los pred son "+str(pred))

    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    print(accuracy(true, pred))

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
    
    longitud = int(len(data)*percentajeTrain)
    train = data[0:longitud]
    test = data[longitud:len(data)]

    return(train, test)

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
    list_newx = list(newx) #recordamos que newx es una sola fila

    for i in range(0, len(data)): 
        list_data = list(data.iloc[i])
        euclideanList.append([euclideanDistance2points(np.array(list_newx), np.array(list_data)), data.mpg[i]]) 
    
    #ordenamos la lista de menor a mayor distancias
    lista_distancias_ordenada = sorted(euclideanList)
    
    #sacamos solo los K primeros más cercanos
    lista_vecinos = lista_distancias_ordenada[0:K]

    for j in range(0, len(lista_vecinos)):
        if lista_vecinos[j][1] == 1:
            cont_1 += 1
        else:
            cont_0 += 1
        
    #asignar clase al nuevo caso
    #return 1 if cont_1>cont_0 else 0
    #añadir nueva columna?????
    if cont_1 > cont_0:
        newx['mpgPred'] = float(1)
    else:
        newx['mpgPred'] = float(0)

    return newx

def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    #np.linalg.norm -> distancia euclidea entre 2 vectores
    return np.linalg.norm(x-y)

# FUNCION accuracy
def accuracy(true, pred):

    return (true/pred)

if __name__ == '__main__':
    np.random.seed(25)
    main()




