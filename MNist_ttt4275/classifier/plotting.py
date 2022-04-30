import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix_heatmap(cf_matrix, title):

    group_counts = ["{0:0.0f}".format(value) for value in np.array(cf_matrix).flatten()]
    group_percentages = np.array([row/np.sum(row) for row in cf_matrix])
    group_percentages = ["{0:.2%}".format(value) for value in group_percentages.flatten()]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(10, 10)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title(f'{title}\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    LABELS = ['0','1','2', '3', '4', '5', '6', '7', '8', '9']
    ax.xaxis.set_ticklabels(LABELS)
    ax.yaxis.set_ticklabels(LABELS)

    ## Display the visualization of the Confusion Matrix.
    plt.show()

def plot_guesses(guesses, title, show_guess=False):
    fig, axarr = plt.subplots(1, len(guesses))
    fig.suptitle(title)
    
    for idx, guess in enumerate(guesses):
        axarr[idx].imshow(np.array(guess.get('image')).reshape((28, 28)), cmap='gray', vmin=0, vmax=255)
        if not show_guess:
            continue
        axarr[idx].set_title(f"Guess: {guess.get('guess')}\nAnswer: {guess.get('answer')}\n")

    plt.show()
