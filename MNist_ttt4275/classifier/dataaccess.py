from mnist import MNIST

mndata = MNIST("../../mnist_data")
mndata.gz = True

def get_training_data():
    return mndata.load_training()

def print_ten_first_images(images):
    for i in range(1, 10):
        print(mndata.display(images[i]))
