from mnist import MNIST

mndata = MNIST("mnist_data")
mndata.gz = True

def get_training_data():
    return mndata.load_training()

def get_testing_data():
    return mndata.load_testing()

def display_image(image):
    return mndata.display(image)
