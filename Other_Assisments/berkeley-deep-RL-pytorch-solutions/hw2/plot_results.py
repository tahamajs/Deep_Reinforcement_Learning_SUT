import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd

Result = namedtuple("Result", "name progress")


def read_data_into_dataframe(filename):
    data = pd.read_json(filename)
    return data


def plot_data(data, value_field, x_axis_field="Train_EnvstepsSoFar"):
    if value_field in data:
        steps_values = data[value_field]
        steps = [sv[0] for sv in steps_values]
        values = [sv[1] for sv in steps_values]
        plt.plot(steps, values, label=value_field)
    plt.xlabel("Step")
    plt.ylabel(value_field)
    plt.legend()
    plt.show()


def read_progress(rootdir):
    import glob

    folders = list(glob.glob(rootdir + "**/**/scalar_data.json", recursive=True))
    print("Found data in: ", folders)
    if len(folders) == 0:
        return []
    data = []
    for folder in folders:
        data.append(read_data_into_dataframe(folder))
    return data


def plot():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--value", type=str, default="Train_AverageReturn")
    args = parser.parse_args()

    data_list = read_progress(args.logdir)
    if len(data_list) == 0:
        print("No data found")
        return

    for data in data_list:
        plot_data(data, args.value)
    plt.savefig(os.path.join(args.logdir, "plot.png"))
    print("Plot saved to", os.path.join(args.logdir, "plot.png"))


if __name__ == "__main__":
    plot()
