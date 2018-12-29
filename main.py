"""
Created on 25/12/2018
@author: nidragedd

This is the main file to use
"""
import os
import argparse
from config import config
from config.config import Configuration
from models.train import trainer
from models.predict import magic_oracle

if __name__ == "__main__":
    # Mandatory arguments to run the program
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Choice between [simple, convnet]")
    ap.add_argument("-o", "--objective", required=True, help="Choice between [train, predict]")
    ap.add_argument("-w", "--working-dir", required=False,
                    help="Use it to set another working directory than the one where this module is executed")
    args = vars(ap.parse_args())

    if args["model"] != trainer.SIMPLE_NN_NAME and args["model"] != trainer.CNN_NAME:
        raise Exception("Model must be a choice between either '{}' or '{}' value".
                        format(trainer.SIMPLE_NN_NAME, trainer.CNN_NAME))
    if args["objective"] != "train" and args["objective"] != "predict":
        raise Exception("Objective must be a choice between either 'train' or 'predict' value")

    if "working-dir" in args:
        working_dir = args["working-dir"]
    else:
        working_dir = os.path.abspath(os.path.dirname(__file__))

    config_dir = os.path.join(working_dir, 'config')
    config.configure_logging(os.path.join(config_dir, 'logging-console.json'))
    # Load program configuration from external JSON format file (assumed to be placed under a 'config' directory
    # within the working directory (specified or computed)
    prog_config = Configuration(os.path.join(config_dir, 'config.json'))
    prog_config.set_working_dir(working_dir)

    if args["objective"] == "train":
        trainer.do_training(prog_config, args["model"])
    elif args["objective"] == "predict":
        magic_oracle.do_magic(prog_config, args["model"])
