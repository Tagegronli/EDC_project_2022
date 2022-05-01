#from Plotscript import *
from Utilities import *


def main_prob1(train, test):
    # Training and testing
    types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]
    steps = [0.01, 0.005, 0.0025, 0.001]
    iterations = 2000
    cm_train_array = []
    cm_test_array = []

    for step in steps:
        [cm_train, cm_test] = simulate_prob1(iterations, step, train, types)
        cm_train_array.append(cm_train)
        cm_test_array.append(cm_test)



    # Plotting MSE
    plt.figure(1)
    plt.grid()
    plt.legend()
    plt.title('MSE for test set at iterations = ' + str(iterations) +
              ',\n training samples = ' + str(train) + ', testing samples = ' + str(test))
    plt.xlabel("Iterations")
    plt.ylabel("MSE")




    # Plotting Error test
    plt.figure(2)
    plt.grid()
    plt.legend()
    plt.title('Error for test set at iterations = '+ str(iterations) +
                 ',\n training samples = ' + str(train) + ', testing samples = ' + str(test))
    plt.xlabel("Iterations")
    plt.ylabel("Error %")


    # Plotting Error training
    plt.figure(3)
    plt.grid()
    plt.legend()
    plt.title('Error for training set at iterations = '+ str(iterations) +
                 ',\n training samples = ' + str(train) + ', testing samples = ' + str(test))
    plt.xlabel("Iterations")
    plt.ylabel("Error %")

    # Code for plotting confusion matrix was found on:
    # https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
    # We use step factor equal 0.005



    # Plottin Confusion matrixes (cm) for traning set.
    plt.figure(4)

    group_counts = ["{0:0.0f}".format(value) for value in cm_train_array[1].flatten()]
    #group_percentages = ["{0:.2%}".format(value) for value in cm_train_array[1].flatten() / np.sum(cm_train_array[1])]
    group_percentages = ["{0:.2%}".format(value) for value in cm_train_array[1].flatten() / train] #30
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)
    ax = sns.heatmap(cm_train_array[1], annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix for traning set @ step factor = ' + str(steps[1]) + ', iterations = '+ str(iterations) +
                 ',\n training samples = ' + str(train) + ', testing samples = ' + str(test) + '\n')
    ax.set_xlabel('\nClassified Category')
    ax.set_ylabel('Actual Flower Category ')
    ax.xaxis.set_ticklabels(types)
    ax.yaxis.set_ticklabels(types)




    # Plottin Confusion matrixes (cm) for test set
    plt.figure(5)

    group_counts = ["{0:0.0f}".format(value) for value in cm_test_array[1].flatten()]
    #group_percentages = ["{0:.2%}".format(value) for value in cm_test_array[1].flatten() / np.sum(cm_test_array[1])]
    group_percentages = ["{0:.2%}".format(value) for value in cm_test_array[1].flatten() / test]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)
    ax = sns.heatmap(cm_test_array[1], annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix for test set @ step factor = ' + str(steps[1]) + ', iterations = '+ str(iterations) +
                 ',\n training samples = ' + str(train) + ', testing samples = ' + str(test) + '\n')
    ax.set_xlabel('\nClassified Flower Category')
    ax.set_ylabel('Actual Flower Category ')
    ax.xaxis.set_ticklabels(types)
    ax.yaxis.set_ticklabels(types)


    # Showing all plots
    plt.show()







main_prob1(30, 20)
main_prob1(20, 30)
#plt.show()