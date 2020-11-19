import numpy as np
import csv
import pandas as pd

path_dataset = "mtcars.csv" # Escoged bien la ruta!!
mtcars = pd.read_csv(path_dataset) # Leemos el csv
d = mtcars.sample(len(mtcars), replace=False)
print(d)