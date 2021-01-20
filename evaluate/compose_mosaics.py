from glob import glob
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from eval_utils import silentremove
from tabulate import tabulate
import pickle
import os, sys, random
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../dataset")
from dataset_utils import LABELS
from cv2 import cv2
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
outpath = "results/thesis_stats_{ckp}/{ckp}_mosaic_{backbone}_{dataset}_{partition}.png"

def create_mosaic(image_paths, predictions, originals, size):
    # rows, columns = size
    # imagex1, imagey1 = 224,224
    # blbx0, blby0 = imagex1+1, imagey1+1
    # blbx1, blby1 = blbx0+200, imagey1
    # grbx0, grby0 =
    # grbx1, grby1 = 
    # for r in rows:
    #     for c in columns:
    #         current_index = r*c
    #         image_path = image_paths[current_index]
    #         image = cv2.imread(image_path)
    #         assert image is not None, "Error loading image {}".format(image_path)

    #         # # TODO add roi selection support
    #         # if not (image_roi is None or avoid_roi):
    #         #   image = cut(image, image_roi)

    #         # values    
            


    #         image = cv2.resize(image, (image_dx, image_dy))
    #         cv2.rectangle(image,(224,0),(200,224),(0,0,0),cv2.FILLED) # predictions box
    #         cv2.rectangle(image,(424,0),(200,224),(0,255,0),cv2.FILLED) # annotations box
    #         cv2.putText(image,str(image_value),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
    #         cv2.rectangle(image,(91,0),(180,35),(0,0,255),cv2.FILLED)
    #         cv2.putText(image,str(image_original_value),(100,25),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)


def convlab(label, task):
    return np.argmax(label) if task != "age" else label

def get_indices(data_dict, image_paths):
    return [data_dict['image_paths'].index(path) for path in image_paths]

def get_path(data_dict, index):
    return data_dict['image_paths'][index]

def get_predictions(data_dict, task, indices):
    return [convlab(data_dict['predictions'][task][index], task) for index in indices]

def get_originals(data_dict, task, indices):
    return [convlab(data_dict['original_labels'][task][index], task) for index in indices]
    

def mosaic(verA, verB, verC, task_list, outpath, size=(4,5)):
    rows, columns = size
    indices = dict()
    predictions = defaultdict(dict)
    originals = dict()
    random_A = np.array(random.sample(list(enumerate(verA['image_paths'])), k=rows*columns))
    image_paths = list()
    indices["verA"] = list()

    for item in random_A:
        indices["verA"].append(int(item[0]))
        image_paths.append(item[1])
    indices["verB"] = get_indices(verB, image_paths)
    indices["verC"] = get_indices(verC, image_paths)

    '''
    image_paths = [path1, path2, path3, pathN]

    predictions = {
        taskX : {
            verY : [value1, value2, value3, valueN]
        }
    }

    After original labels check and compression:
    originals = {
        taskX : [value1, value2, value3, valueN]
    }

    '''
    
    for data_name, data_dict in zip(["verA", "verB", "verC"], [verA, verB, verC]):
        for task in task_list:
            predictions[task][data_name] = get_predictions(data_dict, task, indices[data_name])
            tmp_originals = get_originals(data_dict, task, indices[data_name])
            if task not in originals:
                originals[task] = tmp_originals
            elif originals[task] != tmp_originals:
                raise Exception("Original labels unmatch between {} and its previous version".format(data_name))
    
    # mosaic = create_mosaic(image_paths, predictions, originals, size)
    # cv2.imwrite(outpath, mosaic)
    return outpath

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

if __name__ == "__main__":
    mosaic_subfolder = os.path.split(outpath)[0].format(ckp=ckp_number)
    os.makedirs(mosaic_subfolder, exist_ok=True)
    print("All mosaics in", mosaic_subfolder, "will be overwritten...")

    for backbone in backbones:
        for dataset, task_list in datasets.items():
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
            tmp_outpath = outpath.format(
                ckp=ckp_number,
                dataset=dataset,
                partition=partition,
                backbone=backbone)
            print("\nComposing mosaic for", dataset, backbone, "...")
            mosaic(verA, verB, verC, task_list, tmp_outpath)

