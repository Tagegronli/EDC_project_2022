from math import floor
import numpy as np
import random
import time
from classifier import dataaccess
from sklearn.cluster import KMeans

class Options:
    '''
    Options for classifier
    When use_clustering=True, templates_per_class is number of clusters per class 
    '''
    def __init__(self, number_of_classes, k=11, use_clustering=False, templates_per_class=None, print_guesses=False) -> None:
        self.n_classes = number_of_classes
        self.clustering = use_clustering
        self.templates_per_class = templates_per_class
        self.k = k
        self.print_guesses = print_guesses

class Results:
    def __init__(self, error_rate, confusion_matrix, t) -> None:
        self.error_rate = error_rate
        self.confusion_matrix = confusion_matrix
        self.time = t
    
    def pprint(self):
        print("Time used: %d ms" % self.time)
        print("Error rate: %f" % self.error_rate)
        print("Confusion matrix:\n", self.confusion_matrix)

def get_nearest_neighbours(templates, image, options : Options):
    npimage = np.array(image)
    distances = list()
    for cls_templates in templates:
        if options.clustering:
            selected_templates = cls_templates
        else:
            selected_templates = random.choices(cls_templates, k=options.templates_per_class)
        distances.extend([np.linalg.norm(npimage-np.array(temp)) for temp in selected_templates])
    partitioned = np.argpartition(distances, options.k)
    idx_k_smallest = partitioned[:options.k]
    return np.array([floor(index/options.templates_per_class) for index in idx_k_smallest])

def print_guess(guess, answer, image=None):
    if guess == answer:
        print("======= CORRECT CLASSIFICATION =======")
    else:
        print("====== INCORRECT CLASSIFICATION ======")
    print("Guess :", guess)
    print("Answer:", answer)
    if image != None:
        print(dataaccess.display_image(image).strip())
    print("=======================================")
    print()

def create_clustered_templates(training_data, options : Options):
    templates = [list() for i in range(options.n_classes)]
    for idx, cls in enumerate(training_data):
        model = KMeans(n_clusters=options.templates_per_class)
        model.fit(cls)
        templates[idx] = model.cluster_centers_
        print("Calculated clusters for %d" % idx)
    return templates

def classify(templates, testing_images, testing_labels, options : Options) -> Results:
    t_init = time.time()
    correct_guesses = 0
    displayederrors = 0
    displayedcorrect = 0
    confusion_matrix = [[0]*options.n_classes for i in range(options.n_classes)]
    for idx, image in enumerate(testing_images):
        nearestneighbors = get_nearest_neighbours(templates, image, options)
        guess = np.bincount(nearestneighbors).argmax()
        answer = testing_labels[idx]
        confusion_matrix[answer][guess] += 1
        
        if answer == guess:
            correct_guesses += 1
            if displayedcorrect < 3:
                print_guess(guess, answer, testing_images[idx])
                displayedcorrect += 1
        elif displayederrors < 3:
            print_guess(guess, answer, testing_images[idx])
            displayederrors += 1
        if options.print_guesses:
            print_guess(guess, answer)
            print()

    return Results(
        1-(correct_guesses/len(testing_images)),
        np.matrix(confusion_matrix),
        int((time.time()-t_init)*1000)
        )