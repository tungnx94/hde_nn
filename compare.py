import os
import numpy as np
import matplotlib.pyplot as plt

import utils

config = utils.read_json("cnf_plot.json")

workDir = config["dir"]
model_dirs = utils.list_folder(workDir)

model_loss = {}

for model_dir in model_dirs:
    name = "_".join(model_dir.split("_")[2:-2])

    res_path = os.path.join(workDir, model_dir, "results.json")
    res = utils.read_json(res_path)

    model_loss[name] = res

metrics = config["plots"]
bar_width = 0.4

limit = {"label": 0.5, "angle": 1.3, "accuracy": 100}

def plot_v(metric, chartName):
    fig, ax = plt.subplots()

    ax.set_ylim([0, limit[metric]])
    #ax.get_xaxis().set_visible(False)

    bar_width = 0.5
    index = 0
    for model, loss in model_loss.items():
        index += 1
        val = loss[metric]
        if metric=="accuracy" and val < 1.0:
            val *= 100
        rect = ax.barh(index, val, bar_width, label=model)

        plt.text(index, val, "%.3f" % val, ha='center', va='bottom')

    ax.set_xlabel("models")
    ax.set_ylabel(chartName)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)

    savePath = os.path.join(workDir, chartName)
    fig.savefig(savePath)


def plot_h(metric, chartName):
    fig, ax = plt.subplots()
    ax.set_xlim([0, limit[metric]])

    x = config["models"]
    X = [name.upper() for name in x]
    if metric=="accuracy":
        y = [model_loss[model][metric]*100 for model in x]
    else:
        y = [model_loss[model][metric] for model in x]

    width = 0.4
    ind = np.arange(len(y))

    for i, v in enumerate(y):
        ax.text(v*1.1, i, "%.3f" % v, va='center', fontweight='bold')

    ax.barh(ind, y, width, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(X, minor=False)
    plt.xlabel(chartName)
    plt.ylabel('models')      
    #plt.autoscale()
    plt.tight_layout()

    savePath = os.path.join(workDir, chartName)
    fig.savefig(savePath)


for loss, chartName in metrics.items():
    plot_v(loss, chartName)
    plot_h(loss, chartName)
