import time
import numpy as np
from math import floor
from classifier import dataaccess, cli, classify 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def run():
    
    args = cli.get_arguments()
    args = cli.check_arguments(args)

    t_init = time.time()
    print("Started classifier.")
    
    training_images, training_labels = dataaccess.get_training_data(by_class=False)
    testing_images, testing_labels = dataaccess.get_testing_data()
    
    ######## WITHOUT CLUSTERING ################
    print("Classifying testing images with templates = training images ...")
    accuracy, confusion_matrix, t = classify.classify_images(training_images, training_labels, testing_images, testing_labels, print_guesses=False)
    print("Time used: %d ms" % t)
    print("Error rate: %f" % (1-accuracy))
    print("Confusion matrix:\n", np.matrix(confusion_matrix))
    #############################################
    
    ############## WITH CLUSTERING ##############
    print("Classifying testing images with clustered templates ...")
    error_rate, confusion_matrix, t = classify.classify_with_clusters(testing_images, testing_labels)
    print("Time used: %d ms" % t)
    print("Error rate:", error_rate)
    print("Confusion matrix:\n", confusion_matrix)
    #############################################

    print("Total run time: %d ms" % ((time.time()-t_init)*1000))
    # sys.exit(1)

