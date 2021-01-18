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

def save_plot_confusion_matrix(result_dict, task, filepath):
    array = task_confusion_matrix(result_dict, task)
    labels = ordered_task_labels(task)
    df_cm = pd.DataFrame(array, index=labels, columns=labels)
    df_cm.style.set_properties(**{'text-align': 'center', 'vertical-align': 'center'})
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    heatmap = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16, "va": 'center'}) # font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=90, va='center')
    heatmap.set(xlabel="Predictions", ylabel="Ground truth")
    plt.tight_layout()
    plt.savefig(filepath)
    return array

def mosaic(data, outpath):
    # predicted = result_dict['predictions'][task]
    # original = result_dict['original_labels'][task]
    return ""

# def create_mosaic(reference, out_path, size=(4,5), avoid_roi=False, images_root=None):
#     rows, columns = size    
#     mosaic = None
#     extra_dim = np.array(reference).shape[1]
#     mosaic_items = np.array(random.choices(reference, k=rows*columns))
#     mosaic_items = np.reshape(mosaic_items, (rows, columns, extra_dim))
#     mse = None
#     mae = None

#     for i in range(rows):
#         mosaic_row = None
#         for j in range(columns):
#             mosaic_item = mosaic_items[i][j]

#             image_path = mosaic_item[0] if images_root is None else os.path.join(images_root, mosaic_item[0])
#             image_value = np.round(np.float(mosaic_item[1]), decimals=1)
#             image_original_value = np.round(np.float(mosaic_item[2]), decimals=1) if mosaic_item[2] is not None else None
#             image_roi = mosaic_item[3] if mosaic_item[3] is not None else None
            
#             image = cv2.imread(image_path)
#             assert image is not None, "Error loading image {}".format(image_path)

#             if not (image_roi is None or avoid_roi):
#                 image = cut(image, image_roi)

#             image = cv2.resize(image, (224, 224))
#             cv2.rectangle(image,(0,0),(90,35),(0,0,0),cv2.FILLED)
#             cv2.putText(image,str(image_value),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)

#             if image_original_value is not None:
#                 cv2.rectangle(image,(91,0),(180,35),(0,0,255),cv2.FILLED)
#                 cv2.putText(image,str(image_original_value),(100,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
#                 square_error = (image_original_value-image_value)**2
#                 mse = square_error if mse is None else mse+square_error
#                 absolute_error = abs(image_original_value-image_value)
#                 mae = absolute_error if mae is None else mae+absolute_error

#             mosaic_row = image if mosaic_row is None else np.concatenate((mosaic_row, image), axis=1)
#         mosaic = mosaic_row if mosaic is None else np.concatenate((mosaic,mosaic_row),axis=0)
#     cv2.imwrite(out_path, mosaic)
#     mse = np.round(mse/(rows*columns), decimals=3) if mse is not None else None
#     mae = np.round(mae/(rows*columns), decimals=3) if mae is not None else None
#     return mse, mae

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

    mosaic_subfolder = suboutpath.format(ckp=ckp_number, aim="mosaics")
    os.makedirs(mosaic_subfolder, exist_ok=True)
    print("All mosaics in", mosaic_subfolder, "will be overwritten...")


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
            # mosaic(verA, outpath.format(
            #     ckp=ckp_number,
            #     dataset=dataset,
            #     partition=partition,
            #     backbone=backbone,
            #     version="verA"))
            verB = pickle.load(open(verB_path, "rb"))
            # mosaic(verB, outpath.format(
            #     ckp=ckp_number,
            #     dataset=dataset,
            #     partition=partition,
            #     backbone=backbone,
            #     version="verB"))
            verC = pickle.load(open(verC_path, "rb"))
            # mosaic(verC, outpath.format(
            #     ckp=ckp_number,
            #     dataset=dataset,
            #     partition=partition,
            #     backbone=backbone,
            #     version="verC"))
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
    print("TSTAT path:", tstat_outpath)
    print("CONFUSION MATRIX path:", cm_outpath)