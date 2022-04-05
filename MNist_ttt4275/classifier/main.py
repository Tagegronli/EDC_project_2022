import time
from classifier import dataaccess
from classifier import cli

def run():
    
    args = cli.get_arguments()
    args = cli.check_arguments(args)

    t_init = time.time()
    print("Started program.")
    
    training_images, training_labels = dataaccess.get_training_data()
    dataaccess.print_ten_first_images(training_images)

    print("Total run time: %d ms" % (time.time()-t_init)*1000)
    # sys.exit(1)
