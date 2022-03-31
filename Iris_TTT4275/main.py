#from Plotscript import *
from Utilities import *

# LINK TIL BLOGG:
# https://towardsdatascience.com/a-look-at-the-maths-behind-linear-classification-166e99a9e5fb

# C_k <- Classifier
# t = [t_1, t_2, t_3] <- Target vector
# Ex: [0, 1, 0] <- When we have a case of Iris Versicolor

# L(x) <- Our learning algoritm
# L(x) = sum(w_i * x_i + w_0)
# =>  L(x) = w^T * x + w_0  (vector form)
# ^Takes the sum over ALL classes
# W <- Matrix  Cx(D+1)


# Then take sigmoid function:
# L(x) = sigmoid( L(x) ) = sigmoid ( w^T * x + w_0 )
# L(x) now gives ether 0 or 1 (binary)

# MSE = (1/2)*sum((L_k - t_k)^2) , k = 1,2,3...N
# ^For matrix calculations: (1/2)*sum((L_k - t_k)^T * (g_k - t_k))

# Take gradient of MSE:
# Grad(MSE) = sum(k=1, N)[(L_k - t_k ) * L_k * (1-L_k)] * x^T

# W(m) = W(m-1) - alpha * grad(MSE)
# m is the iteration of W

# C = number of classes
# D

step = 0.5

types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]


setosa_array = get_flower_array("class_1.csv", types[0])
versicolor_array = get_flower_array("class_1.csv", types[1])
virginica_array = get_flower_array("class_1.csv", types[2])




W = make_zero_matrix(len(types), len(setosa_array))


# TESTING
#######################################
#data = ["2.2","3.1","4.2","5.3", "sunflower"]
#sunflower = flower(data)
#w = get_flower_array("class_1.csv", types[1])
#print(w[5].type)


def Train(W, step):

    return W

######################################