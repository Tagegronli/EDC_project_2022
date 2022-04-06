from math import floor
import numpy as np
import random
import time
from classifier import dataaccess
from sklearn.cluster import KMeans

def get_nearest_neighbours(templates, template_labels, k, image, maxcheck=10000):
    npimage = np.array(image)
    selected_template_indices = random.choices(range(len(templates)-1), k=maxcheck)
    distances = [np.linalg.norm(npimage-np.array(templates[index])) for index in selected_template_indices]
    partitioned = np.argpartition(distances, k)
    k_smallest = partitioned[:k]
    return np.array([template_labels[selected_template_indices[index]] for index in k_smallest])

def make_guess(templates, template_labels, k, image, maxcheck=10000):
    nearest_neighbors = get_nearest_neighbours(templates, template_labels, k, image, maxcheck=maxcheck)
    guess = np.bincount(nearest_neighbors).argmax()
    return guess

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

def classify_images(templates, template_labels, images, images_labels, print_guesses=False):
    t_init = time.time()
    K = 11
    NUMCLASSES = 10
    confusion_matrix = [[0]*NUMCLASSES for i in range(NUMCLASSES)]
    correct_guesses = 0
    displayederrors = 0
    displayedcorrect = 0
    for index in range(len(images)):
        guess = make_guess(templates, template_labels, K, images[index], maxcheck=100)
        fasit = images_labels[index]
        confusion_matrix[fasit][guess] += 1
        if fasit == guess:
            correct_guesses += 1
            if displayedcorrect < 3:
                print_guess(guess, fasit, images[index])
                displayedcorrect += 1
            
        elif displayederrors < 3:
            print_guess(guess, fasit, images[index])
            displayederrors += 1

        if print_guesses:
            print_guess(guess, fasit)
            print()
    return (correct_guesses/len(images), confusion_matrix, int((time.time()-t_init)*1000))

def classify_with_clusters(testing_images, testing_labels):
    NUMCLASSES = 10

    training_data = dataaccess.get_training_data(by_class=True)
    templates = create_clustered_templates(training_data)

    # classify testing_images
    t_init = time.time() # timing just the classifying
    corrects = 0
    confusion_matrix = [[0]*NUMCLASSES for i in range(NUMCLASSES)]
    for idx, image in enumerate(testing_images):
        nearestneighbors = get_nearest_neighbours_clustered(templates, 7, image, cluster_per_class=64)
        guess = np.bincount(nearestneighbors).argmax()
        answer = testing_labels[idx]
        confusion_matrix[answer][guess] += 1
        if guess == answer:
            corrects += 1

    return (1-(corrects/len(testing_images)), np.matrix(confusion_matrix), int((time.time()-t_init)*1000))

def create_clustered_templates(training_data):
    CLUSTERS_PER_CLASS = 64
    NUMCLASSES = 10
    templates = [list() for i in range(NUMCLASSES)]
    for idx, cls in enumerate(training_data):
        model = KMeans(n_clusters=CLUSTERS_PER_CLASS)
        model.fit(cls)
        templates[idx] = model.cluster_centers_
        print("Calculated clusters for %d" % idx)
    return templates


def get_nearest_neighbours_clustered(templates, k, image, cluster_per_class=64):
    npimage = np.array(image)
    distances = list()
    for cls_templates in templates:
        distances.extend([np.linalg.norm(npimage-np.array(temp)) for temp in cls_templates])    
    partitioned = np.argpartition(distances, k)
    idx_k_smallest = partitioned[:k]
    return np.array([floor(index/cluster_per_class) for index in idx_k_smallest])