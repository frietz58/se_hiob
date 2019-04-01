import os
import argparse
import matplotlib
matplotlib.use('Agg')  # for plot when display is undefined
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

parser = argparse.ArgumentParser(description="")
parser.add_argument("-ptr", "--pathresults", help="")
args = parser.parse_args()


def is_hiob_exe(file):
    return os.path.isdir(file) and "hiob-execution" in file


def create_inner_outer_csv():
    files_in_dir = os.listdir(args.pathresults)

    hiob_executions = []
    for file in files_in_dir:
        if is_hiob_exe(os.path.join(args.pathresults, file)):
            hiob_executions.append(os.path.join(args.pathresults, file))

    df = pd.DataFrame(columns=["Inner punish threshold", "Outer punish threshold", "Total success", "Total precision", "Average size score"])
    for hiob_execution in hiob_executions:

        # get inner outer
        with open(os.path.join(hiob_execution, "tracker.yaml"), "r") as stream:
            try:
                configuration = yaml.safe_load(stream)
                scale_estimator_conf = configuration["scale_estimator"]
            except yaml.YAMLError as exc:
                print(exc)

            inner = configuration["scale_estimator"]["inner_punish_threshold"]
            outer = configuration["scale_estimator"]["outer_punish_threshold"]

        # get prec succ
        if not os.path.exists(os.path.join(hiob_execution, "evaluation.txt")):
            print("didnt find evaluation.txt in " + str(hiob_execution) + " skipping")
            continue

        with open(os.path.join(hiob_execution, "evaluation.txt"), "r") as eval_txt:
            lines = eval_txt.readlines()
            for line in lines:
                line = line.replace("\n", "")
                key_val = line.split("=")
                if key_val[0] == "total_precision_rating":
                    total_prec = float(key_val[1])
                elif key_val[0] == "total_success_rating":
                    total_succ = float(key_val[1])

        # get size values for each sequence
        files_in_execution = os.listdir(hiob_execution)
        sequences_in_execution = []
        for file in files_in_execution:
            if "tracking" and os.path.isdir(os.path.join(hiob_execution, file)):
                sequences_in_execution.append(os.path.join(hiob_execution, file))

        curr_execution_size_scores = []
        for sequence in sequences_in_execution:
            with open(os.path.join(sequence, "evaluation.txt"), "r") as seq_eval_txt:
                lines = seq_eval_txt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    if key_val[0] == "area_between_size_curves":
                        curr_execution_size_scores.append(float(key_val[1]))

        average_size_score = np.around(sum(curr_execution_size_scores) / len(curr_execution_size_scores), decimals=3)

        row = {"Inner punish threshold": inner,
               "Outer punish threshold": outer,
               "Total success": total_succ,
               "Total precision": total_prec,
               "Average size score": average_size_score}

        df = df.append(row, ignore_index=True)

    csv_path = os.path.join(args.pathresults, "inner_outer_opt.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


def create_3d_scatter(csv_path, z_axis, filename, z_lim_max):
    fig = plt.figure()
    ax = Axes3D(fig)

    df = pd.read_csv(csv_path)

    sequence_containing_x_vals = list(df["Inner punish threshold"])
    sequence_containing_y_vals = list(df["Outer punish threshold"])
    sequence_containing_z_vals = list(df[str(z_axis)])

    color = np.abs(sequence_containing_z_vals)

    ax.set_xlim3d(0.1, 1.0)
    ax.set_ylim3d(0.1, 1.0)

    if z_lim_max == "dynamic":
        z_lim_max = max(sequence_containing_z_vals) + 5
        print(max(sequence_containing_z_vals))

    ax.set_zlim3d(0.1, z_lim_max)

    ax.set_xlabel('Inner punish threshold')
    ax.set_ylabel('Outer punish threshold')
    ax.set_zlabel(z_axis)

    cmhot = plt.get_cmap("viridis")
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=color, cmap=cmhot, vmax=z_lim_max)

    ax.view_init(30, 160)

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
    #     print(ax.azim)

    plt.savefig(os.path.join(os.path.dirname(csv_path), filename))
    plt.show()


def create_3d_surf(csv_path):
    fig = plt.figure()
    ax = Axes3D(fig)

    df = pd.read_csv(csv_path)

    x = list(df["Inner punish threshold"])
    y = list(df["Outer punish threshold"])
    z = list(df["Total success"])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # as plot_surface needs 2D arrays as input
    x = np.arange(10)
    y = np.array(range(10, 15))
    # we make a meshgrid from the x,y data
    X, Y = np.meshgrid(x, y)
    # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
    Z = np.asarray([np.asarray(z), np.asarray(z)])

    # data_value shall be represented by color
    data_value = np.random.rand(len(y), len(x))
    # map the data to rgba values from a colormap
    colors = cm.ScalarMappable(cmap="viridis").to_rgba(data_value)

    # plot_surface with points X,Y,Z and data_value as colors
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                           linewidth=0, antialiased=True)

    plt.savefig(os.path.join(os.path.dirname(csv_path), "3d_inner_outer_surf"))
    plt.show()


def main():
    inner_outer_csv = create_inner_outer_csv()
    create_3d_scatter(csv_path=inner_outer_csv,
                      z_axis="Total success",
                      filename="inner_outer_success",
                      z_lim_max=1.0)

    create_3d_scatter(csv_path=inner_outer_csv,
                      z_axis="Average size score",
                      filename="inner_outer_size",
                      z_lim_max=100)

    # create_3d_surf(csv_path=inner_outer_csv)


if __name__ == "__main__":
    main()
