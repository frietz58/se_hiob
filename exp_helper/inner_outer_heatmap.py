import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import yaml

parser = argparse.ArgumentParser(description="")
parser.add_argument("-ptr", "--pathresults", help="")
args = parser.parse_args()

if args.pathresults is None:
    args.pathresults = "/informatik2/students/home/5rietz/BA/new_formula_opt"


def heatmap(data,
            row_labels,
            col_labels,
            ax=None,
            cbar_kw={},
            cbarlabel="",
            xlabel="",
            ylabel="",
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel(xlabel)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(ylabel)

    return im, cbar


def annotate_heatmap(im,
                     data=None,
                     valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None,
                     **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
        data.reshape([data.shape[0], data.shape[1]])

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            if data[i, j] == np.max(data):
                kw.update(color="r")
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


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

        # if inner == 0.6 and outer == 1.0:
        #     print()

        print("inner: {}, outer {}, success: {}".format(inner, outer, total_succ))

    csv_path = os.path.join(args.pathresults, "inner_outer_opt.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


if __name__ == "__main__":

    # create csv from the results in the given folder
    opt_results_csv = create_inner_outer_csv()

    # create df from created csv
    df = pd.read_csv(opt_results_csv)
    inner_thresh_vals = np.unique(df["Inner punish threshold"])
    outer_thresh_vals = np.unique(df["Outer punish threshold"])
    success_scores = np.asarray(df["Total success"])
    precision_scores = np.asarray(df["Total precision"])
    size_error = np.asarray(df["Average size score"])

    inner = np.arange(0.1, 1.1, 0.1)
    inner = [str(np.around(number, decimals=3)) for number in inner]

    outer = np.arange(0.1, 1.1, 0.1)
    outer = [str(np.around(number, decimals=3)) for number in outer]

    data = np.random.rand(np.shape(inner)[0], np.shape(outer)[0])
    # data = [str(np.around(number, decimals=3)) for number in data]

    fig, ax = plt.subplots()

    # im, cbar = heatmap(data, inner, outer, ax=ax, cmap="YlGn_r", cbarlabel="success")
    im, cbar = heatmap(success_scores.reshape([10, 10]), row_labels=inner_thresh_vals, col_labels=outer_thresh_vals,
                       ax=ax, cmap="YlGn_r", cbarlabel="success", xlabel="Outer threshold", ylabel="Inner threshold")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()