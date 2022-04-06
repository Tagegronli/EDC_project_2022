from math import floor
import numpy as np
import random
import time
from classifier import dataaccess

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

def guess_images(templates, template_labels, images, images_labels, print_guesses=False):
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
    return (correct_guesses/len(images), confusion_matrix)