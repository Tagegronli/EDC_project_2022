from Iris_TTT4275.Utilities import *

#Sepal width has most overlap

def prob2(train, test):
    # Training and testing
    figure_index = 1
    types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]
    steps = [0.01, 0.005, 0.0025, 0.001]
    iterations = 5000


    # [flower.sepal_l, flower.sepal_w, flower.petal_l, flower.petal_w]
    features_array = [[1,1,1,1], [1,0,1,1], [0,0,1,1], [0,0,0,1]]
    for features in features_array:
        cm_train_array = []
        cm_test_array = []

        for step in steps:
            [cm_train, cm_test] = simulate_prob2(iterations, step, train, types, features, figure_index)
            cm_train_array.append(cm_train)
            cm_test_array.append(cm_test)



        #Plotting Error
        plt.figure(figure_index)
        plt.grid()
        plt.legend()
        plt.title('Error rate for test set with iterations = '+ str(iterations) +
                     ',\n' + str(features) + " = [Sepal lenght, Sepal width, Petal lenght, Petal width]")
        plt.xlabel("Iterations")
        plt.ylabel("Error %")

        figure_index +=1



        # We use step factor equal 0.005

        # Plottin Confusion matrixes (cm) for test set
        plt.figure(figure_index)

        group_counts = ["{0:0.0f}".format(value) for value in cm_test_array[1].flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm_test_array[1].flatten() / test]
        labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(3, 3)
        ax = sns.heatmap(cm_test_array[1], annot=labels, fmt='', cmap='Blues')
        ax.set_title('Confusion Matrix for test set @ step factor = ' + str(steps[1]) + ', iterations = '+ str(iterations) +
                     ',\n'+ str(features) + " = [Sepal lenght, Sepal width, Petal lenght, Petal width]")
        ax.set_xlabel('\nClassified Flower Category')
        ax.set_ylabel('Actual Flower Category ')
        ax.xaxis.set_ticklabels(types)
        ax.yaxis.set_ticklabels(types)

        figure_index +=1
    plt.show()



 # [flower.sepal_l, flower.sepal_w, flower.petal_l, flower.petal_w]
#prob2(30,20)


