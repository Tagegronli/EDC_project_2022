from mnist import MNIST

mndata = MNIST("mnist_data")
mndata.gz = True

def get_training_data(by_class=False):
    images, labels = mndata.load_training()
    if not by_class:
        return images, labels
    
    cls_training_sets = [list() for i in range(10)]
    for index, label in enumerate(labels):
        cls_training_sets[label].append(images[index])
    return cls_training_sets

def get_testing_data():
    return mndata.load_testing()

def display_image(image):
    return mndata.display(image)
