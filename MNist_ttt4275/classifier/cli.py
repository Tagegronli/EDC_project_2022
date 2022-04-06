import argparse
from classifier import main


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-fan",
            "--full-argument-name",
            dest="full_argument_name",
            # action="store_true",
            default=False,
            help="explanation")

    args = parser.parse_args()
    return args


def check_arguments(args):
    # do some check if arguments are reasonable, 
    # maybe raise error if not?
    somearg = args.full_argument_name
    if somearg:
        return somearg
