from Iris_TTT4275.Utilities import *
import random


def histogram():
    types = ["Iris setosa", "Iris Versicolor", "Iris Virginica"]

    BASE = "Iris_TTT4275/"
    setosa_array = get_flower_array(f"{BASE}class_1.csv", types[0])
    versicolor_array = get_flower_array(f"{BASE}class_2.csv", types[1])
    virginica_array = get_flower_array(f"{BASE}class_3.csv", types[2])

    setosa_sepal_l = []
    setosa_sepal_w = []
    setosa_petal_l = []
    setosa_petal_w = []
    for f in setosa_array:
        setosa_sepal_l.append(f.sepal_l)
        setosa_sepal_w.append(f.sepal_w)
        setosa_petal_l.append(f.petal_l)
        setosa_petal_w.append(f.petal_w)


    versicolor_sepal_l = []
    versicolor_sepal_w = []
    versicolor_petal_l = []
    versicolor_petal_w = []
    for f in versicolor_array:
        versicolor_sepal_l.append(f.sepal_l)
        versicolor_sepal_w.append(f.sepal_w)
        versicolor_petal_l.append(f.petal_l)
        versicolor_petal_w.append(f.petal_w)


    virginica_sepal_l = []
    virginica_sepal_w = []
    virginica_petal_l = []
    virginica_petal_w = []
    for f in virginica_array:
        virginica_sepal_l.append(f.sepal_l)
        virginica_sepal_w.append(f.sepal_w)
        virginica_petal_l.append(f.petal_l)
        virginica_petal_w.append(f.petal_w)

    (fig, ax) = plt.subplots(2,2)
    bins = np.linspace(4.0, 8.0, 40)

    ax[0,0].hist(setosa_sepal_l, bins, alpha=0.5, label='Setosa')
    ax[0,0].hist(versicolor_sepal_l, bins, alpha=0.5, label='Versicolor')
    ax[0,0].hist(virginica_sepal_l, bins, alpha=0.5, label = 'Virginica')
    ax[0,0].set_title("Sepal lenght")
    ax[0,0].legend()

    bins = np.linspace(1.5, 4.5, 30)
    ax[0,1].hist(setosa_sepal_w, bins, alpha=0.5, label='Setosa')
    ax[0,1].hist(versicolor_sepal_w, bins, alpha=0.5, label='Versicolor')
    ax[0,1].hist(virginica_sepal_w, bins, alpha=0.5, label='Virginica')
    ax[0,1].set_title("Sepal width")
    ax[0,1].legend()


    bins = np.linspace(0.5, 7.0, 65)
    ax[1,0].hist(setosa_petal_l, bins, alpha=0.5, label='Setosa')
    ax[1,0].hist(versicolor_petal_l, bins, alpha=0.5, label='Versicolor')
    ax[1,0].hist(virginica_petal_l, bins, alpha=0.5, label = 'Virginica')
    ax[1,0].set_title("Petal lenght")
    ax[1,0].legend()



    bins = np.linspace(0.0, 3.0, 30)
    ax[1,1].hist(setosa_petal_w, bins, alpha=0.5, label='Setosa')
    ax[1,1].hist(versicolor_petal_w, bins, alpha=0.5, label='Versicolor')
    ax[1,1].hist(virginica_petal_w, bins, alpha=0.5, label = 'Virginica')
    ax[1,1].set_title("Petal width")
    ax[1,1].legend()


    plt.show()


#hisogram()