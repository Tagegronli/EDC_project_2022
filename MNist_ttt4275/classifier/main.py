import time
from classifier import dataaccess, classify, plotting

def run():    
    t_init = time.time()
    print("Started classifier.")
    NCLASSES = 10
    
    training_data = dataaccess.get_training_data(by_class=True)
    testing_images, testing_labels = dataaccess.get_testing_data()

    # NN without clustering (1a)
    opts = classify.Options(
        NCLASSES, 
        k=1,
        use_clustering=False,
        templates_per_class=30)
    templates = classify.get_random_templates(training_data, opts)
    print("Classifying testing images with templates = training images ...")
    results = classify.classify(templates, testing_images, testing_labels, opts)
    results.pprint()
    
    plotting.confusion_matrix_heatmap(results.confusion_matrix, 
    f"Confusion matrix for NN ({opts.templates_per_class} stochastically selected templates per class)")
    # Plotting 3 misclassified pictures (1b)
    plotting.plot_guesses(results.incorrect_samples, "", show_guess=True)
    # Plotting 3 correctly classified pictures (1c)
    plotting.plot_guesses(results.correct_samples, "")


    opts = classify.Options(
        NCLASSES,
        k=1,
        use_clustering=True, 
        templates_per_class=64)
    # Perform clustering (2a)
    templates = classify.create_clustered_templates(training_data, opts)
    # NN with clustering (2b)
    print("Classifying testing images with clustered templates ...")
    results = classify.classify(templates, testing_images, testing_labels, opts)
    results.pprint()

    plotting.confusion_matrix_heatmap(results.confusion_matrix, 
    f"Confusion matrix for NN ({opts.templates_per_class} clusters per class, k = {opts.k})")
    plotting.plot_guesses(results.correct_samples, "")
    plotting.plot_guesses(results.incorrect_samples, "", show_guess=True)

    # kNN with clustering (2c)
    opts = classify.Options(
        NCLASSES,
        k=7,
        use_clustering=True, 
        templates_per_class=64)
    results = classify.classify(templates, testing_images, testing_labels, opts)
    results.pprint()

    plotting.confusion_matrix_heatmap(results.confusion_matrix,
    f"Confusion matrix for kNN ({opts.templates_per_class} clusters per class, k = {opts.k})")

    # ALL TASKS COMPLETED.
    print("Total run time: %d ms" % ((time.time()-t_init)*1000))
    # sys.exit(1)

