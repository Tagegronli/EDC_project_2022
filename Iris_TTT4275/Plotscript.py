import matplotlib.pyplot as plt
import csv

def plot_objects(filename, color, name, plt1, plt2):
    class_ = csv.reader(open(filename))

    #Converting data to matrix
    class_matrix = []
    for obj in class_:
        class_matrix.append(obj)


    #plotting points with labels (so that we can display the label of 1 point)
    plt1.plot(float(class_matrix[0][0]), float(class_matrix[0][1]), marker="o", color=color, label=name)
    plt1.legend() #Showing only label of 1 point so we dont get 1000 labels

    plt2.plot(float(class_matrix[0][2]), float(class_matrix[0][3]), marker="o", color=color, label=name)
    plt2.legend() #Showing only label of 1 point so we dont get 1000 labels



    #plotting the rest of the points without labels
    for obj in class_matrix[1:]:
        plt1.plot(float(obj[0]), float(obj[1]), marker="o", color=color) #, label = "AAA")
        plt2.plot(float(obj[2]), float(obj[3]), marker="o", color=color) #, label = "AAA")


    return


def plotscript():
    #setting names for parameters
    naming = ["sepal length [cm]", "sepal width [cm]", "petal length [cm]", "petal width [cm]"]

    #making subplots
    fig, (ax1, ax2) = plt.subplots(2)



    #Naming
    #plt1.set_title("Sepal length vs width")
    #plt2.set_title("Petal length vs width")
    plt.setp(ax1, xlabel=naming[0], ylabel = naming[1])
    plt.setp(ax2, xlabel=naming[2], ylabel = naming[3])



    #Plotting
    ax1.grid()
    ax2.grid()
    plot_objects("class_1.csv", "red", "Setosa", ax1, ax2)
    plot_objects("class_2.csv", "green", "Versicolor", ax1, ax2)
    plot_objects("class_3.csv", "blue", "Virginica", ax1, ax2)
    plt.show()

    return



plotscript()