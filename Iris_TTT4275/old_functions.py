
# Adding x is not 5x1 required in the compendium
def update_x(x):
    x = np.append(x, 1.0)
    #print(x)
    return x


# W is now 3x5 as required in the compendium
def update_W(W, w_0):
    W = np.append([W[0], W[1], W[2], W[3]], [w_0], axis=0)  # <- Had to hardcode this...
    #print(W)
    return W