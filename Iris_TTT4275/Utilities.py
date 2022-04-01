import numpy as np
import csv
import math

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
# JUST USE: np.zeros(C, D)


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


def compute_gi(x, W, w_0):
    gi = np.add(np.dot(W, x), w_0)
    return gi

def gi_sigmoid(x, W, w_0):
    zi = compute_gi(x, W, w_0)
    gi = []
    for z in zi:
        gi.append(1 / (1 + math.exp(-z)))

    return gi