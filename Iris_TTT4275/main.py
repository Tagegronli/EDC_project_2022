#from Plotscript import *
import numpy as np
from Utilities import *


##########################################################################################
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
##########################################################################################



def main():
    # Training and testing
    run = [0.01, 0.005, 0.0025, 0.001]
    for i in run:
        simulate(2000, i)



    # Plotting
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


main()