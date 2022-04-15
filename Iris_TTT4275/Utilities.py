import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import seaborn as sns


# Flower class, holds all data and has a "type" which is the label.
# If type is not spesified, it will try to look for it in the dataset
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





# Makes array of flower objects, we can decide type if we want (not necessary if it exist in data)
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




# Makes 3x5 zero matrix. Initialising W with w_0
def init_W(C, D):
    return np.zeros(shape = (D+1, C))





def init_training_data(types, samples):
    setosa_array = get_flower_array("class_1.csv", types[0])
    versicolor_array = get_flower_array("class_2.csv", types[1])
    virginica_array = get_flower_array("class_3.csv", types[2])

    training_data = setosa_array[:samples] + versicolor_array[:samples] + virginica_array[:samples]

    return training_data




def init_test_data(types, samples):
    setosa_array = get_flower_array("class_1.csv", types[0])
    versicolor_array = get_flower_array("class_2.csv", types[1])
    virginica_array = get_flower_array("class_3.csv", types[2])

    test_data = setosa_array[samples:] + versicolor_array[samples:] + virginica_array[samples:]

    return test_data




def get_x_data(traning_data):
    x_data = []
    for i in range(len(traning_data)):
        x = [traning_data[i].sepal_L, traning_data[i].sepal_w, traning_data[i].petal_l, traning_data[i].petal_w]
        x_data.append(x)

    return x_data




def init_x(flower):
    x = [1, flower.sepal_L, flower.sepal_w, flower.petal_l, flower.petal_w]
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




# Shown in (22)
def compute_gradMSE_k(x, g, t):
    a = []
    for i in range(len(g)):
        b = (g[i] - t[i]) * g[i] * (1 - g[i])
        a.append(b)
    #print(res)
    #print(np.transpose([x]))
    gradMSE_k = np.dot(np.transpose([x]),[a])

    return gradMSE_k





def compute_gradMSE(types, W, traning_data): # W already updated
    gradMSE = np.zeros(shape = (4+1,3))
    for k in traning_data:
        x = init_x(k)
        g = g_sigmoid(x, W)
        t = get_t(k, types)

        gradMSE_k = compute_gradMSE_k(x, g, t)
        gradMSE = np.add(gradMSE, gradMSE_k)

    return gradMSE



def iterate_W(W, gradMSE, step):
    W = np.subtract(W, step*gradMSE)
    return W


def compute_MSE(test_data, W, types):
    array = []
    for f in test_data:
        x = init_x(f)
        t = get_t(f, types)
        g = g_sigmoid(x, W)
        MSE = np.dot(np.transpose(np.subtract(g, t)), np.subtract(g, t))
        array.append(MSE)
        #print(MSE)

    MSE = 0
    for o in array:
        MSE += o

    return (1/2)*MSE

def get_results(test_data, W, types):
    correct = 0
    false = 0
    for f in test_data:
        t = get_t(f, types)
        g = g_sigmoid(init_x(f), W)
        classified = np.argmax(g)
        true  = np.argmax(t)
        if(classified == true):
            correct +=1
        else:
            false += 1

    return [correct, false]



##### WORK IN PROGRESS ######
def get_result_matrix(data, W, types):
    result_matrix = np.zeros((3,3), dtype=int)
    #header = ["Classified / True class", types[0], types[1], types[2]]
    #empty = [".", ".", ".", ".", "."]
    #result_matrix.append(header)
    #for i in range(5):
    #    result_matrix.append(empty)

    for f in data:
        t = get_t(f, types)
        g = g_sigmoid(init_x(f), W)

        classified = np.argmax(g)
        true = np.argmax(t)
        result_matrix[true][classified] += 1






    return result_matrix

# Returns confusion matrix to traning and test data and plots MSE and error for each step factor
def simulate(iterations, step, train, test, types):
    #print("Initialsing variables")
    traning_data = init_training_data(types, train)    # Flower array used for training
    test_data = init_test_data(types, test)           # Flower array used for testing

    W = init_W(3, 4)



    print("Step size = " + str(step) + ", processing... ")
    iteration = np.arange(0, iterations)
    MSE_array = []
    error_array = []

    for i in iteration:
        # Training
        gradMSE = compute_gradMSE(types, W, traning_data)
        W = iterate_W(W, gradMSE, step)

        #Testing
        MSE = compute_MSE(test_data, W, types)
        MSE_array.append(MSE)

        results = get_results(test_data, W, types)

        error_array.append(100*results[1]/(results[0] + results[1]))


    # Plotting MSE and Error
    plt.figure(1)
    plt.plot(iteration, MSE_array, label = "step factor =" +str(step))
    MSE_min = np.argmin(MSE_array)
    print("Smallest MSE: " + str(MSE_array[MSE_min]) + " at iteration: " + str(MSE_min) + ", step size = " + str(step))



    # Plotting Error
    plt.figure(2)
    plt.plot(iteration, error_array, label="step factor = " + str(step))
    error_min = np.argmin(error_array)
    print("Smallest error: " + str(error_array[MSE_min]) + " at iteration: " + str(error_min) + ", step size = " + str(step))
    print(" ")


### WIP:

    # Plotting confusion matrix
    cm_train = get_result_matrix(traning_data, W, types)
    cm_test = get_result_matrix(test_data, W, types)

    #print("Result traning set: ")
    #print(cm_train)
    #print("Result test set: ")
    #print(cm_test)

    return [cm_train, cm_test]





