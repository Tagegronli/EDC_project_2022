import numpy as np
import csv

class flower:
    def __init__(self, data, type = None): # <- Must take in data only consisting of strings
        self.sepal_L = float(data[0])
        self.sepal_w = float(data[1])
        self.petal_l = float(data[2])
        self.petal_w = float(data[3])

        if(type != None):
            self.type = type

        elif(len(data) > 4):
            self.type = data[4]

        else:
            self.type = "unknown"

        return


# Makes a CxD with only zeros
def make_zero_matrix(C, D):
    W = []
    for i in range(D):
        W.append(np.zeros(C))

    return W


# Makes array of flower objects, we can decide type IF we want (not necessary if it exist in data)
def get_flower_array(filename, type = None):
    data_lines = csv.reader(open(filename))
    flower_array = []

    if(type == None):
        for line in data_lines:
            flower_array.append(flower(line))
    else:
        for line in data_lines:
            flower_array.append(flower(line, type))

    return flower_array


