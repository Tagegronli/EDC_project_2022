import time
import numpy as np
import random
from classifier import dataaccess, cli, classify 

def run():
    
    args = cli.get_arguments()
    args = cli.check_arguments(args)

    t_init = time.time()
    print("Started classifier.")
    
    training_images, training_labels = dataaccess.get_training_data()
    testing_images, testing_labels = dataaccess.get_testing_data()
    
    print("Classifying testing images with templates = training images ...")
    accuracy, confusion_matrix = classify.guess_images(training_images, training_labels, testing_images, testing_labels)
    print("Confusion matrix:\n", np.matrix(confusion_matrix))
    print("Error rate: %f" % (1-accuracy))

    print("Total run time: %d ms" % ((time.time()-t_init)*1000))
    # sys.exit(1)
