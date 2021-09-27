# Please run this script at the root dir of cords

import os
# import pickle
import pickle5 as pickle
import numpy as np
import itertools
import json
import time

from scripts.run_plot import plot

if __name__ == '__main__':
    results_path = os.path.join("./", "scripts", "RESULTS")

    # Organize dataloader
    for f in os.listdir(results_path):
        save_file = os.path.join(results_path, f, "save_dict.pickle")
        if not f.startswith("EXP_") or (not os.path.exists(save_file)):
            continue
        with open(save_file, 'rb') as handle:
            save_dict = pickle.load(handle)
        training_args = vars(save_dict["args"])
        if training_args["dss_strategy"] == "glister":
            print(f)
