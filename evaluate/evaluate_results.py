from glob import glob
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from eval_utils import silentremove
from tabulate import tabulate
import pickle
import os, sys
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
sys.path.append("../dataset")
from dataset_utils import LABELS

############## PARAMS ################
ckp_number = sys.argv[1]
######################################

datasets = {
    "VGGFace2" : ["gender", "age", "ethnicity"], 
    "RAF-DB" : ["emotion"],
    "LFWPlus" : ["gender", "age"],
    "FairFace" : ["ethnicity"]
}
partition = "test"
backbones = ["mobilenetv3", "resnet50", "seresnet50"]
versions = ["verA", "verB", "verC"]
inpath = "results/thesis_results_{ckp}/results_{dataset}_{partition}_of_checkpoint_{ckp}_from_net{backbone}_versionver[ABC]_*"
outpath = "results/thesis_stats_{ckp}/{ckp}_{aim}_{target}.txt"
suboutpath = "results/thesis_stats_{ckp}/{aim}"

def uncategorize(task, x, y):
    if task != "age":
        return np.argmax(x, axis=-1), np.argmax(y, axis=-1)
    return np.array(x), np.array(y)
    
def task_confusion_matrix(result_dict, task):
    predicted = result_dict['predictions'][task]
    original = result_dict['original_labels'][task]
    predicted, original = uncategorize(task, predicted, original)
    return confusion_matrix(original, predicted, normalize='true')

def squeeze_label(label, task):
    if task == "emotion":
        return label[:3]
    elif task == "ethnicity":
        return "".join([s[0] for s in label.split()])
    elif task == "gender":
        return label.capitalize()
    else:
        return label

def ordered_task_labels(task):
    return [squeeze_label(duple[0], task) for duple in sorted(LABELS[task].items(), key=lambda item:item[1])]

def fontsize(task):
    return {"gender" : 56,
            "ethnicity": 26,
            "emotion": 20}.get(task)

def save_plot_confusion_matrix(result_dict, task, filepath):
    array = task_confusion_matrix(result_dict, task)
    labels = ordered_task_labels(task)
    df_cm = pd.DataFrame(array, index=labels, columns=labels)
    df_cm.style.set_properties(**{'text-align': 'center', 'vertical-align': 'center'})
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    size = fontsize(task)
    heatmap = sn.heatmap(df_cm, annot=True, annot_kws={"size": size, "va": 'center'}) # font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=90, va='center')
    heatmap.set(xlabel="Predictions", ylabel="Ground truth")
    plt.tight_layout()
    plt.savefig(filepath)
    return array


def accuracy_score(predictions, originals, thresholds):
    scores = dict()
    for th in thresholds:
        counter = 0
        for pred, orig in zip(predictions, originals):
            error = abs(pred-orig)
            if error >= th:
                counter += 1
        scores[th] = 100 * (1 - counter/len(predictions))
    return scores

def save_plot_cumulative_score(result_dict, task, filepath):
    predictions = result_dict['predictions'][task]
    originals = result_dict['original_labels'][task]
    thresholds = [t for t in range(1, 11)]
    scores = accuracy_score(predictions, originals, thresholds)
    ys = list(scores.values())
    _, ax = plt.subplots()
    plt.plot([t-1 for t in thresholds], ys, '-gD')
    plt.grid()
    ax.set_xticks(np.arange(0, len(thresholds), step=1))
    ax.set_xticklabels(thresholds)
    ax.set_yticks(np.arange(0, 101, step=10))
    ax.set_xlabel('Error tolerance (years)')
    ax.set_ylabel('Cumulative score (%)')
    ax.set(facecolor = "#cfd8dc")
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    return scores


def statistic(v1, v2, metric):
    if metric == "tstat":
        return ttest_ind(v1, v2).pvalue
    elif metric == "cohen":
        return cohen_kappa_score(np.rint(v1).astype(int), np.rint(v2).astype(int))
    else:
        raise Exception("{} is not a supported metric.".format(metric))
 

def run_stat(task, verA, verB, verC, name="", metric="tstat"):
    verAoriginals, verApredicted = uncategorize(task, verA['original_labels'][task], verA['predictions'][task])
    verBoriginals, verBpredicted = uncategorize(task, verB['original_labels'][task], verB['predictions'][task])
    verCoriginals, verCpredicted = uncategorize(task, verC['original_labels'][task], verC['predictions'][task])
    results = dict()
    results["A-self"] = statistic(verAoriginals, verApredicted, metric)
    results["B-self"] = statistic(verBoriginals, verBpredicted, metric)
    results["C-self"] = statistic(verCoriginals, verCpredicted, metric)
    results["A-B"] = statistic(verApredicted, verBpredicted, metric)
    results["B-A"] = statistic(verBpredicted, verApredicted, metric)
    results["A-A"] = statistic(verApredicted, verApredicted, metric)
    results["C-B"] = statistic(verCpredicted, verBpredicted, metric)
    results["B-C"] = statistic(verBpredicted, verCpredicted, metric)
    results["B-B"] = statistic(verBpredicted, verBpredicted, metric)
    results["C-A"] = statistic(verCpredicted, verApredicted, metric)
    results["A-C"] = statistic(verApredicted, verCpredicted, metric)
    results["C-C"] = statistic(verCpredicted, verCpredicted, metric)
    line0 = [name, "Truth", "verA", "verB", "verC"]
    line1 = ["verA", results["A-self"], results["A-A"], results["A-B"], results["A-C"]]
    line2 = ["verB", results["B-self"], results["B-A"], results["B-B"], results["B-C"]]
    line3 = ["verC", results["C-self"], results["C-A"], results["C-B"], results["C-C"]]
    return [line0, line1, line2, line3]

if __name__ == "__main__":
    tstat_outpath = outpath.format(ckp=ckp_number, aim="tstat", target="all")
    silentremove(tstat_outpath)
    print("Removed TSTAT old file:", tstat_outpath)

    cohen_outpath = outpath.format(ckp=ckp_number, aim="cohen", target="all")
    silentremove(cohen_outpath)
    print("Removed COHEN old file:", cohen_outpath)

    # agreement_outpath = outpath.format(ckp=ckp_number, aim="agreement", target="all")
    # silentremove(agreement_outpath)
    # print("Removed AGREEMENT old file:", agreement_outpath)

    cm_outpath = outpath.format(ckp=ckp_number, aim="confusion_matrix", target="all")
    silentremove(cm_outpath)
    print("Removed CONFUSION MATRIX old file:", cm_outpath)

    cm_plot_subfolder = suboutpath.format(ckp=ckp_number, aim="confusion_matrix")
    os.makedirs(cm_plot_subfolder, exist_ok=True)
    print("All confusion matrices in", cm_plot_subfolder, "will be overwritten...")

    cs_plot_subfolder = suboutpath.format(ckp=ckp_number, aim="cumulative_score")
    os.makedirs(cs_plot_subfolder, exist_ok=True)
    print("All cumulative scores in", cs_plot_subfolder, "will be overwritten...")

    for backbone in backbones:
        for dataset in datasets.keys():
            print("\nOpening", dataset, backbone, "results...")
            verA_path, verB_path, verC_path = sorted(glob(inpath.format(
                dataset=dataset,
                partition=partition,
                backbone=backbone,
                ckp=ckp_number
            )))
            verA = pickle.load(open(verA_path, "rb"))
            verB = pickle.load(open(verB_path, "rb"))
            verC = pickle.load(open(verC_path, "rb"))
            for task in datasets[dataset]:
                title = "{bb}_{task}_{dataset}_{partition}".format(
                    bb=backbone,
                    task=task,
                    dataset=dataset,
                    partition=partition
                )
                print("Running", task, "T-STAT...")
                tstat_table = run_stat(task, verA, verB, verC, title, metric="tstat")
                tab = tabulate(tstat_table, headers="firstrow", tablefmt="grid", numalign="right", stralign="center")
                with open(tstat_outpath, "a") as fp:
                    fp.write(tab+"\n")

                print("Running", task, "COHEN KAPPA SCORE...")
                cohen_table = run_stat(task, verA, verB, verC, title, metric="cohen")
                tab = tabulate(cohen_table, headers="firstrow", tablefmt="grid", numalign="right", stralign="center")
                with open(cohen_outpath, "a") as fp:
                    fp.write(tab+"\n")

                # print("Running", task, "AGREEMENT...")
                # agreement_table = run_stat(task, verA, verB, verC, title, metric="agreement")
                # tab = tabulate(agreement_table, headers="firstrow", tablefmt="grid", numalign="right", stralign="center")
                # with open(agreement_outpath, "a") as fp:
                #     fp.write(tab+"\n")
                
                if task != "age":
                    print("Evaluating", task, "confusion matrix...")
                    verA_cm = save_plot_confusion_matrix(verA, task, os.path.join(cm_plot_subfolder, title+"_verA.png"))
                    verB_cm = save_plot_confusion_matrix(verB, task, os.path.join(cm_plot_subfolder, title+"_verB.png"))
                    verC_cm = save_plot_confusion_matrix(verC, task, os.path.join(cm_plot_subfolder, title+"_verC.png"))
                    with open(cm_outpath, "a") as fp:
                        fp.write("\nConfusion Matrix version A " + title + "\n")
                        fp.write(tabulate(verA_cm,tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")+"\n")
                        fp.write("\nConfusion Matrix version B " + title + "\n")
                        fp.write(tabulate(verB_cm,tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")+"\n")
                        fp.write("\nConfusion Matrix version C " + title + "\n")
                        fp.write(tabulate(verC_cm,tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")+"\n")
                else:
                    print("Evaluating", task, "cumulative score...")
                    verA_cs = save_plot_cumulative_score(verA, task, os.path.join(cs_plot_subfolder, title+"_verA.png"))
                    print("Version A cumulative score", verA_cs)
                    verB_cs = save_plot_cumulative_score(verB, task, os.path.join(cs_plot_subfolder, title+"_verB.png"))
                    print("Version B cumulative score", verB_cs)
                    verC_cs = save_plot_cumulative_score(verC, task, os.path.join(cs_plot_subfolder, title+"_verC.png"))
                    print("Version C cumulative score", verC_cs)


    print("TSTAT path:", tstat_outpath)
    print("CONFUSION MATRIX path:", cm_outpath)