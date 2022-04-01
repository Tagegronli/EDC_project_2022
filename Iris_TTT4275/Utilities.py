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



# W is now 3x5 as required in the compendium
def update_W(W, w_0):
    W = np.append([W[0], W[1], W[2], W[3]], [w_0], axis=0)  # <- Had to hardcode this...
    #print(W)
    return W



# x is not 5x1 required in the compendium
def update_x(x):
    x = np.append(x, 1.0)
    #print(x)
    return x


# Computes g from W and w_0 as shown in (7)
def compute_g(x, W):
    #print(W)
    #print(x)
    g = np.dot(np.transpose(W),x)
    return g




# computes sigmoid of g elementwise and returns "new" g. Shown in (20)
def g_sigmoid(x, W):
    z = compute_g(x, W)
    g = []
    for zi in z:
        g.append(1 / (1 + math.exp(-zi)))

    return g


# Returns correct t array: [1,0,0], [0,1,0], [0,0,1]
def get_t(flower, types):
    type = flower.type
    if(type == types[0]):
        return [1,0,0]
    elif(type == types[1]):
        return [0,1,0]
    elif(type == types[2]):
        return [0,0,1]

    else:
        print("No type found, returning zero-array")
        return [0, 0, 0]



##### WORK IN PROGRESS ######


# Shown in (22)
def compute_divMSE_k(x, g, t):
    comp = []
    for i in range(len(g)):
        s = (g[i] - t[i]) * g[i] * (1 - g[i])
        comp.append(s)
    #print(res)
    #print(np.transpose([x]))
    divMSE_k = np.dot(np.transpose([x]),[comp])

    return divMSE_k


def compute_divMSE(W, training_data): # W already updated
    for f in training_data:
        x = update_x([f.sepal_L, f.sepal_w, f.petal_l, f.petal_w])
        g = g_sigmoid(w)
    return 0



