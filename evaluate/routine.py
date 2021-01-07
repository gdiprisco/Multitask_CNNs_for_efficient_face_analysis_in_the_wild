import os
from glob import glob
import pickle
from tabulate import tabulate
from eval_utils import silentremove

############## PARAMS ################ #TODO argparse
GPU = 1
ckp_number = 600
only_inference_time = False
inference_time = False

outpath = "results/thesis_results_{ckp}".format(ckp=ckp_number)
pathlist = sorted(glob(os.path.join("/user/gdiprisco/multitask/thesis_trained", "*"))) 
datasets = ["VGGFace2", "RAF-DB", "LFWPlus", "FairFace"]
partitions = ["test"] #, "train", "val"]
checkpoint = "checkpoint.{ckp}.hdf5".format(ckp=ckp_number)
######################################


command = "python3 evaluate_model.py --dataset {dataset} --partition {partition} --outpath {outpath} --gpu {gpu} --path {path}"

pickle_results = os.path.join(outpath, "summary_results.pk")
tabulate_results_path = os.path.join(outpath, "tabulate_summary_results.txt")
tabulate_task_specific_results_path = os.path.join(outpath, "tabulate_task_specific_summary_results.txt")

if not only_inference_time:
    silentremove(pickle_results)
    silentremove(tabulate_results_path)
    silentremove(tabulate_task_specific_results_path)

pickle_stats = os.path.join(outpath, "summary_stats.pk")
tabulate_stats_path = os.path.join(outpath, "tabulate_summary_statistics.txt")

if only_inference_time or inference_time:
    silentremove(pickle_stats)
    silentremove(tabulate_stats_path)

for dataset in datasets:
    for partition in partitions:
        for path in pathlist:
            model_path = os.path.join(path, checkpoint)
            formatted_command = command.format(
                                            dataset=dataset,
                                            partition=partition,
                                            outpath=outpath,
                                            gpu=GPU,
                                            path=model_path
                                        )
            if only_inference_time:
                formatted_command += " --only_inference_time"
            if inference_time:
                formatted_command += " --inference_time"
            os.system(formatted_command)


# Summary results
summary_results = []
with open(pickle_results, "rb") as pkf:
    try:
        while True:
            summary_results.extend(pickle.load(pkf))
    except EOFError:
        pass

general_tabulate_head = ["Model", "Version", "Checkpoint", "Dataset", "Partition"]
results_header = general_tabulate_head + ["Task", "Metric", "Value"]
print("\nTabulating summary results...")
tab = tabulate(summary_results, results_header, tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")
print(tab)
with open(tabulate_results_path, "a") as fp:
    fp.write(tab+"\n")
print("\nResult tabulated with no error.")

for task in ["Gender", "Age", "Ethnicity", "Emotion"]:
    for dataset in datasets:
        tmp_sum = list()
        for element in summary_results:
            if element[3] == dataset and element[5] == task:
                tmp_sum.append(element)
        if tmp_sum:
            tab = tabulate(tmp_sum, results_header, tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")
            with open(tabulate_task_specific_results_path, "a") as fp:
                fp.write(tab+"\n")


# Statistics
if only_inference_time or inference_time:
    summary_stats = []
    with open(pickle_stats, "rb") as pkf:
        try:
            while True:
                summary_stats.extend(pickle.load(pkf))
        except EOFError:
            pass

    results_header = general_tabulate_head + ["Inference time (secs)", "FPS", "Parameters count", "Memory usage (bytes)"]
    print("\nTabulating summary statistics...")
    tab = tabulate(summary_stats, results_header, tablefmt="grid", numalign="right", stralign="center", floatfmt=".5f")
    with open(tabulate_stats_path, "a") as fp:
        fp.write(tab+"\n")
    print("\nStatistics tabulated with no error.")

