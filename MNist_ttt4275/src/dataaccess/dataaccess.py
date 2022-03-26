from mnist import MNIST

mndata = MNIST("../../mnist_data")
mndata.gz = True

training_images, training_labels = mndata.load_training()

print(training_images[0])
print(training_labels[0])


print(mndata.display(training_images[0]))