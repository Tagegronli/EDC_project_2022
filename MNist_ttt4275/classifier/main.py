import time
from classifier import dataaccess, cli, classify

def run():    
    args = cli.get_arguments()
    args = cli.check_arguments(args)

    t_init = time.time()
    print("Started classifier.")
    NCLASSES = 10
    
    training_data = dataaccess.get_training_data(by_class=True)
    testing_images, testing_labels = dataaccess.get_testing_data()
    
    # Without clustering (Part 1)
    opts = classify.Options(
        NCLASSES, 
        k=11, 
        use_clustering=False, 
        templates_per_class=30)
    print("Classifying testing images with templates = training images ...")
    results = classify.classify(training_data, testing_images, testing_labels, opts)
    results.pprint()

    # With clustering (Part 2)
    print("Classifying testing images with clustered templates ...")
    opts = classify.Options(
        NCLASSES,
        k=7,
        use_clustering=True, 
        templates_per_class=64)
    templates = classify.create_clustered_templates(training_data, opts)
    results = classify.classify(templates, testing_images, testing_labels, opts)
    results.pprint()

    print("Total run time: %d ms" % ((time.time()-t_init)*1000))
    # sys.exit(1)

