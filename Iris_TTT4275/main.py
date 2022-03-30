from Plotscript import *
import numpy as np

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

class flower:
    def __init__(self, data):
        # Instance Variables:
        self.sepal_L = data[0]
        self.sepal_w = data[1]
        self.petal_l = data[2]
        self.petal_w = data[3]

        if(len(data) > 4):
            self.type = data[4]
        else:
            self.type = "unknown"

        return


def compare():
    C = 3
    D = 5

    t = [0, 0, 0]
    W = []
    for i in range(D+1):
        W.append(np.zeros(C))



data = [2,3,4,5, "sunflower"]

sunflower = flower(data)
print(sunflower.type)
print(sunflower.petal_w)
print(sunflower.petal_l)