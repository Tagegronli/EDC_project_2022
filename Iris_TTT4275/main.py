#from Plotscript import *
import numpy as np

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

step = 0.5

types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]


setosa_array = get_flower_array("class_1.csv", types[0])
versicolor_array = get_flower_array("class_1.csv", types[1])
virginica_array = get_flower_array("class_1.csv", types[2])
traning_data = setosa_array + versicolor_array + virginica_array




#x = [setosa_array[0].sepal_L, setosa_array[0].sepal_w, setosa_array[0].petal_l, setosa_array[0].petal_w]
#t = get_t(setosa_array[0], types)


# Need to be initialized:
W = np.zeros((4,3)) # 3x4
w_0 = [1,1,1]

W = update_W(W, w_0) # W is now 3x5 as required in the compendium
#x = update_x(x) # x is not 5x1 required in the compendium



# TESTING
#######################################

#g = compute_g(x, W)
g = g_sigmoid(x, W)


print(len(traning_data))
#print(g)
#print(compute_divMSE_k(x, g, t))
#print(compute_g(x, W, w_0))
#print(W)
#print(g_sigmoid(x, W, w_0))



######################################