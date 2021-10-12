# Please run this script at the root dir of cords

import os
# import pickle
import pickle5 as pickle
import numpy as np
import itertools
import json
import time

from scripts.run_plot import plot

_metrics_first_k = "args"
_metrics_second_k = "dss_strategy"
_metrics_plot_x = ["epochs", "train_elapsed"]
_metrics_plot_y = ["train_loss", "train_accu", "valid_loss", "valid_accu"]
_metrics_avg = ["test_loss", "test_accu"]

_metrics_list = _metrics_plot_x + _metrics_plot_y + _metrics_avg

if __name__ == '__main__':
    results_path = os.path.join("./", "scripts", "RESULTS")
    exp_metrics = {}
    first_k, first_k_obj, second_k = [], [], []

    start = time.time()
    save_path = os.path.join(".", "scripts", "PLOTS_%s" % start)
    os.makedirs(save_path)

    # Organize data
    for f in os.listdir(results_path):
        save_file = os.path.join(results_path, f, "save_dict.pickle")
        if not f.startswith("EXP_") or (not os.path.exists(save_file)):
            continue
        with open(save_file, 'rb') as handle:
            save_dict = pickle.load(handle)

        print((save_dict["args"].select_ratio, save_dict["args"].dataset, save_dict["args"].dss_strategy))
        # print(f)
        # print(save_dict)
        # print("--------------------------------------------")

        training_args = vars(save_dict[_metrics_first_k])

        _first_k_obj = {_k: _v for (_k, _v) in training_args.items() if _k != _metrics_second_k}
        _first_k = json.dumps(_first_k_obj,
                              sort_keys=True, indent=4)
        _second_k = training_args[_metrics_second_k]

        if _first_k not in first_k:
            first_k.append(_first_k)
            first_k_obj.append(_first_k_obj)

        if _second_k not in second_k:
            second_k.append(_second_k)

        if _first_k not in exp_metrics:
            exp_metrics[_first_k] = {}

        if _second_k not in exp_metrics[_first_k]:
            exp_metrics[_first_k][_second_k] = {_metric: [] for _metric in _metrics_list}

        for _metric in _metrics_list:
            exp_metrics[_first_k][_second_k][_metric] = save_dict[_metric]

    plot_comb = list(itertools.product(_metrics_plot_x, _metrics_plot_y))

    # Plot all
    avg_metric = {}
    for _first_k in first_k:
        avg_metric[_first_k] = {}
        for _second_k in exp_metrics[_first_k]:
            avg_metric[_first_k][_second_k] = {}
            for _metric_avg in _metrics_avg:
                if _metric_avg not in exp_metrics[_first_k][_second_k]:
                    continue
                avg_metric[_first_k][_second_k][_metric_avg] = np.mean(exp_metrics[_first_k][_second_k][_metric_avg])

    for (_x_metric, _y_metric) in plot_comb:
        for (_first_k, _first_k_obj) in zip(first_k, first_k_obj):
            x_list, y_list, labels = [], [], []
            for _second_k in exp_metrics[_first_k]:
                if (_first_k not in exp_metrics) or (_second_k not in exp_metrics[_first_k]):
                    continue
                print(_second_k)
                labels.append(_second_k)
                # x_list.append(exp_metrics[_first_k][_second_k][_x_metric][:100])
                # y_list.append(exp_metrics[_first_k][_second_k][_y_metric][:100])
                x_list.append(exp_metrics[_first_k][_second_k][_x_metric])
                y_list.append(exp_metrics[_first_k][_second_k][_y_metric])
            # Disable avg_metric it for the time
            # note_obj = {"args": _first_k_obj, "avg_metric": avg_metric[_first_k]}
            note_obj = {"args": _first_k_obj}
            # plot(x_list, y_list, labels=labels, xlabel=_x_metric, ylabel=_y_metric,
            #      note=json.dumps(note_obj, sort_keys=True, indent=4), legend=True)
            # filename = "%s_%s_%s_%s_%s.png" % (_first_k_obj["dataset"], _x_metric, _y_metric,
            #                                    "adaptive: %s" % _first_k_obj["is_adaptive"],
            #                                    _first_k_obj["model"])
            filename = "%s_%s_%s_%s_%s_%s.png" % (_first_k_obj["dataset"], _x_metric, _y_metric,
                                                  "adaptive: %s" % _first_k_obj["is_adaptive"],
                                                  _first_k_obj["model"] if "model" in _first_k_obj else "None",
                                                  _first_k_obj["select_ratio"])
            plot(x_list, y_list, labels=labels, xlabel=_x_metric, ylabel=_y_metric,
                 note=json.dumps(note_obj, sort_keys=True, indent=4), legend=True,
                 save_path=os.path.join(save_path, filename))
