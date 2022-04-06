import time
import numpy as np
from math import floor
from classifier import dataaccess, cli, classify 

from sklearn.cluster import KMeans
from sklearn.datasets import make_classification

def run():
    
    args = cli.get_arguments()
    args = cli.check_arguments(args)

    t_init = time.time()
    print("Started classifier.")
    
    training_images, training_labels = dataaccess.get_training_data(by_class=False)
    testing_images, testing_labels = dataaccess.get_testing_data()

    CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    ######## WITHOUT CLUSTERING ################
    # print("Classifying testing images with templates = training images ...")
    # accuracy, confusion_matrix = classify.classify_images(training_images, training_labels, testing_images, testing_labels, print_guesses=True)
    # print("Confusion matrix:\n", np.matrix(confusion_matrix))
    # print("Error rate: %f" % (1-accuracy))
    #############################################
    
    ############## WITH CLUSTERING ##############
    templates = [list() for i in range(10)]
    training_data = dataaccess.get_training_data(by_class=True)
    for idx, cls in enumerate(training_data):
        model = KMeans(n_clusters=64)
        #print(cls[0])
        model.fit(cls)
        templates[idx] = model.cluster_centers_
        print("Calculated clusters for %d" % idx)
        # print(model.cluster_centers_[63])
    

    corrects = 0
    NUMCLASSES = 10
    confusion_matrix = [[0]*NUMCLASSES for i in range(NUMCLASSES)]
    for idx, image in enumerate(testing_images):
        nearestneighbors = get_nearest_neighbours_clustered(templates, 7, image, cluster_per_class=64)
        guess = np.bincount(nearestneighbors).argmax()
        answer = testing_labels[idx]
        confusion_matrix[answer][guess] += 1
        classify.print_guess(guess, answer)
        if guess == answer:
            corrects += 1

    # accuracy, confusion_matrix = classify.classify_images(training_images, training_labels, testing_images, testing_labels, print_guesses=True)
    print("Error rate:", 1-(corrects/len(testing_images)))
    print("Confusion matrix:")
    print(np.matrix(confusion_matrix))

    #############################################

    print("Total run time: %d ms" % ((time.time()-t_init)*1000))
    # sys.exit(1)

def get_nearest_neighbours_clustered(templates, k, image, cluster_per_class=64):
    npimage = np.array(image)
    distances = list()
    for cls_templates in templates:
        distances.extend([np.linalg.norm(npimage-np.array(temp)) for temp in cls_templates])    
    partitioned = np.argpartition(distances, k)
    idx_k_smallest = partitioned[:k]
    return np.array([floor(index/cluster_per_class) for index in idx_k_smallest])

def make_guess(templates, template_labels, k, image, maxcheck=10000):
    nearest_neighbors = get_nearest_neighbours_clustered(templates, template_labels, k, image, maxcheck=maxcheck)
    guess = np.bincount(nearest_neighbors).argmax()
    return guess