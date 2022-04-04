#from Plotscript import *
import numpy as np
import matplotlib.pyplot as plt
from Utilities import *

# LINK TIL BLOGG:
# https://towardsdatascience.com/a-look-at-the-maths-behind-linear-classification-166e99a9e5fb

# C_k <- Classifier
# t = [t_1, t_2, t_3] <- Target vector
# Ex: [0, 1, 0] <- When we have a case of Iris Versicolor

# g(x) <- Our learning algoritm
# g(x) = sum(w_i * x_i + w_0)
# =>  g(x) = w^T * x + w_0  (vector form)
# ^Takes the sum over ALL classes
# W <- Matrix  Cx(D+1)


# Then take sigmoid function:
# g(x) = sigmoid( L(x) ) = sigmoid ( w^T * x + w_0 )
# g(x) now gives ether 0 or 1 (binary)

# MSE = (1/2)*sum((L_k - t_k)^2) , k = 1,2,3...N
# ^For matrix calculations: (1/2)*sum((L_k - t_k)^T * (g_k - t_k))

# Take gradient of MSE:
# Grad(MSE) = sum(k=1, N)[(L_k - t_k ) * L_k * (1-L_k)] * x^T

# W(m) = W(m-1) - alpha * grad(MSE)
# m is the iteration of W


# C = 3 (number of classes)
# D = 4 (data point)
# W = CxD -> 3x4 matrix (3 rows, 4 columns)

# DATA = ["sepal length [cm]", "sepal width [cm]", "petal length [cm]", "petal width [cm]"]


######################################


def main(iterations, step):
    #print("Initialsing variables")
    types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]
    traning_data = init_training_data(types, 30)    # Flower array used for training
    test_data = init_test_data(types, 20)           # Flower array used for testing

    W = init_W(3, 4)



    print("Training and testing, " + "step size = " + str(step))
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


    #Plotting MSE and Error
    plt.figure(1)
    plt.plot(iteration, MSE_array, label = "step: "+str(step))
    MSE_min = np.argmin(MSE_array)
    print("Smallest MSE: " + str(MSE_array[MSE_min]) + " at iteration: " + str(MSE_min) + ", step size = " + str(step))

    plt.figure(2)
    plt.plot(iteration, error_array, label = "step: "+str(step))
    error_min = np.argmin(error_array)
    print("Smallest error: " + str(error_array[MSE_min]) + " at iteration: " + str(error_min) + ", step size = " + str(step))
    print(" ")



# Plotting
run = [0.01, 0.005, 0.0025, 0.001]
for i in run:
    main(2000, i)

plt.figure(1)
plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")

plt.figure(2)
plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Error %")

plt.show()
