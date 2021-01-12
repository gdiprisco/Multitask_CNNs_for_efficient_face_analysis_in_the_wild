import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras, sys, os, re, argparse, time, csv, keras, pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
from eval_utils import *
from memory_usage import keras_model_memory_usage_in_bytes

sys.path.append("../dataset")
sys.path.append("../training")

# from dataset_tools import cut
# from antialiasing import BlurPool
# from mobile_net_v2_keras import relu6
from model_abstract import age_relu
# from train import loss, loss_weights
# from train import accuracy as accuracy_metrics
from train import get_versioned_metrics
from model_MobileNet_v3_build import HSigmoid, Hswish
from dataset_utils import readcsv, writecsv
from dataset_VGGFace2_VMER_VMAGE import Vgg2DatasetMulti
from dataset_RAF import RAFDBMulti
from dataset_LFWplus import LFWPlusMulti
from dataset_FairFace import FairFaceMulti

available_datasets = {
    "VGGFace2" : {
        "dataset" : Vgg2DatasetMulti, 
        "metrics" : {
            "gender" : categorical_accuracy,
            "age" : mae_accuracy,
            "ethnicity" : categorical_accuracy,
            "emotion" : None 
        }
    },
    "RAF-DB" : {
        "dataset" : RAFDBMulti,
        "metrics" : {
            "gender" : categorical_accuracy,
            "age" : None, #raf_age_groups_accuracy,
            "ethnicity" : None, #raf_race_groups_accuracy,
            "emotion" : categorical_accuracy 
        },
    },
    "LFWPlus" : {
        "dataset" : LFWPlusMulti,
        "metrics" : {
            "gender" : categorical_accuracy,
            "age" : mae_accuracy,
            "ethnicity" : None,
            "emotion" : None 
        },
    },
    "FairFace" : {
        "dataset" : FairFaceMulti,
        "metrics" : {
            "gender" : None, #categorical_accuracy,
            "age" : None, #fairface_age_groups_accuracy,
            "ethnicity" : categorical_accuracy,
            "emotion" : None
        }
    }
}


parser = argparse.ArgumentParser(description='Dataset evaluation, provided for train, val and test partition')
parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset', choices=list(available_datasets.keys()))
parser.add_argument('--partition', dest='partition', type=str, help='Source path of HDF5 model to test', choices=['train', 'val', 'test'])
parser.add_argument('--path', dest='inpath', type=str, help='Source path of HDF5 model to test')
parser.add_argument('--gpu', dest="gpu", type=str, default="0", help="Gpu to use")
parser.add_argument('--outpath', dest="outpath", type=str, default="results", help='Destination path of results file')
parser.add_argument('--only_inference_time', action="store_true", help="Only inference time test will be executed")
parser.add_argument('--inference_time', action="store_true", help="Also inference time test will be executed")
parser.add_argument('--checkpoint', dest="checkpoint", type=int, help="Specify checkpoint if 'path' is the main directory of the model")
args = parser.parse_args()

custom_objects = {
                #   'BlurPool': BlurPool,
                #   'relu6': relu6,
                'age_relu': age_relu,
                'Hswish': Hswish,
                'HSigmoid': HSigmoid
}

def get_model_info(filepath):
    model_dirname = os.path.split(os.path.split(filepath)[0])[1]
    version = re.search("versionver[ABC]", model_dirname)
    if not version:
        raise Exception("Unable to infer model version from path splitting")
    version = version[0].replace("version", "")
    modelname = re.search("_net[A-Za-z0-9]*", model_dirname)
    if not modelname:
        raise Exception("Unable to infer model name from path splitting")
    modelname = modelname[0].replace("_net", "")
    checkpoint = os.path.split(filepath)[1].split(".")[1]
    return modelname, version, model_dirname, checkpoint

def load_keras_model(filepath):
    _, version, _, _ = get_model_info(filepath)
    loss, loss_weights, accuracy_metrics, _ = get_versioned_metrics(version)
    model = keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
    model.compile(loss=loss, loss_weights=loss_weights, optimizer='sgd', metrics=accuracy_metrics)
    INPUT_SHAPE = (112, 112, 3)
    return model, INPUT_SHAPE

def get_filepath_ck(main_filepath, checkpoint):
    tail = 'checkpoint.{}.hdf5'.format(str(checkpoint).zfill(2))
    return os.path.join(main_filepath, tail)


def check_prediction(image_paths, predictions, original_labels, image_rois):
    lp, lo = None, None
    for k, vlist in predictions.items():
        if lp is None:
            lp = len(vlist)
        elif lp != len(vlist) :
            raise Exception("Unmatching predictions for {}: {} vs. {} annotations".format(k, lp, len(vlist)))

    for k, vlist in original_labels.items():
        if lo is None:
            lo = len(vlist)
        elif lo != len(vlist) :
            raise Exception("Unmatching labels for {}: {} vs. {} annotations".format(k, lo, len(vlist)))
    
    if not len(image_paths) == lp == lo == len(image_rois):
            details = "Images: {} - Predictions: {} - Originals: {} - Rois: {}".format(len(image_paths), lp, lo, len(image_rois))
            raise Exception("Invalid prediction on batch\n", details)

def run_test(Dataset, modelpath, batch_size=64, partition='test'):
    model, INPUT_SHAPE = load_keras_model(modelpath)
    dataset = Dataset(partition=partition,
                      target_shape=INPUT_SHAPE,
                      augment=False,
                      preprocessing='vggface2',
                      age_annotation="number",
                      include_gender=True,
                      include_age_group=True,
                      include_race=True)
    data_gen = dataset.get_generator(batch_size, fullinfo=True)
    image_paths = list()
    image_rois = list()
    original_labels = defaultdict(list)
    predictions = defaultdict(list)
    for batch in tqdm(data_gen):
        res = model.predict(batch[0])
        predictions["gender"].extend(res[-4])
        predictions["age"].extend([age[0] for age in res[-3]])
        predictions["ethnicity"].extend(res[-2])
        predictions["emotion"].extend(res[-1])
        original_labels["gender"].extend(batch[1][0])
        original_labels["age"].extend(batch[1][1])
        original_labels["ethnicity"].extend(batch[1][2])
        original_labels["emotion"].extend(batch[1][3])
        image_paths.extend(batch[2])
        image_rois.extend(batch[3])
    check_prediction(image_paths, predictions, original_labels, image_rois)
    GPU_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=batch_size)
    print(" --- TEST RUNNED ---")
    print("Memory usage {} bytes".format(GPU_bytes))
    print(" -------------------")
    return image_paths, predictions, original_labels, image_rois


def run_inference_time_test(Dataset, modelpath, partition='test'):
    model, INPUT_SHAPE = load_keras_model(modelpath)
    dataset = Dataset(partition=partition,
                      target_shape=INPUT_SHAPE,
                      augment=False,
                      preprocessing='vggface2',
                      age_annotation="number",
                      include_gender=True,
                      include_age_group=True,
                      include_race=True)
    data_gen = dataset.get_generator(batch_size=1, fullinfo=True)
    print("Dataset batches %d" % len(data_gen))
    start_time = time.time()
    # _ = model.evaluate_generator(data_gen, verbose=1, workers=4)
    for batch in tqdm(data_gen):
        _ = model.predict(batch[0])
    spent_time = time.time() - start_time
    batch_average_time = spent_time / len(data_gen)
    fps = 1/batch_average_time
    print("Evaluate time %d s" % spent_time)
    print("Batch time %.10f s, FPS: %.3f" % (batch_average_time, fps))
    GPU_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=1)
    print(" --- INFERENCE TEST RUNNED ---")
    print("Memory usage {} bytes".format(GPU_bytes))
    print(" -----------------------------")
    return batch_average_time, fps, GPU_bytes, model.count_params()


def zip_reference(image_paths, predictions, original_labels, image_rois):
    reference = list()
    predictions_reference = list()
    original_reference = list()
    for g, a, et, em in zip(predictions["gender"], predictions["age"], predictions["ethnicity"], predictions["emotion"]):
        predictions_reference.append((g, a, et, em))
    for g, a, et, em in zip(original_labels["gender"], original_labels["age"], original_labels["ethnicity"], original_labels["emotion"]):
        original_reference.append((g, a, et, em))
    for path, pred, original, roi in zip(image_paths, predictions_reference, original_reference, image_rois):
        reference.append((path, pred, original, roi))
    return reference

def _refactor_data(data):
    restored = list()
    for item in data:
        if item[3] is not None:
            roi = [np.int(x) for x in item[3:7]]
            item  = [item[0], np.float(item[1]), np.float(item[2]), roi]
        restored.append(item)
    return restored


def evaluate_metrics(predictions, originals, metric):
    result = dict()
    for task in ["gender", "age", "ethnicity", "emotion"]:
        task_metric = metric[task]
        result[task] = task_metric(predictions[task], originals[task]) if task_metric is not None else None
    return result["gender"], result["age"], result["ethnicity"], result["emotion"]
    

if '__main__' == __name__:
    if not args.inpath.endswith('.hdf5'): raise Exception("Only .hdf5 files are supported.")
    start_time = datetime.today()
    os.makedirs(args.outpath, exist_ok=True)
    Dataset = available_datasets[args.dataset]["dataset"]
    
    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    
    modelname, version, model_dirname, checkpoint = get_model_info(args.inpath)
    out_path = "results_{}_{}_of_checkpoint_{}_from{}".format(args.dataset, args.partition, checkpoint, model_dirname)
    out_path = os.path.join(args.outpath, out_path)
    general_summary_data = [modelname.upper(), version, checkpoint, args.dataset, args.partition]
    
    if args.only_inference_time:
        print("Only inference time will be calculated!")
    else:
        pk_out_path = out_path+".pk"
        if os.path.exists(pk_out_path):
            print("LOADING CACHED RESULTS...")
            pk_data = pickle.load(open(pk_out_path, "rb"))
            predictions = pk_data["predictions"]
            original_labels = pk_data["original_labels"]
        else:
            print("Running test from scratch ...")
            image_paths, predictions, original_labels, image_rois  = run_test(Dataset, args.inpath, batch_size=64, partition=args.partition)
            print("Writing Pickle", pk_out_path, "...")
            pk_data = {
                "image_paths" : image_paths,
                "predictions" : predictions,
                "original_labels" : original_labels,
                "image_rois" : image_rois
            }
            pickle.dump(pk_data, open(pk_out_path, "wb"))
            # #################### CSV ####################
            # print("Managing data ...")
            # reference = zip_reference(image_paths, predictions, original_labels, image_rois)
            # csv_out_path = out_path+".csv"
            # print("Writing CSV", csv_out_path, "...")
            # writecsv(csv_out_path, reference) # TODO better format CSV
            # #############################################
        print("Evaluating metrics of", modelname, version, "for", args.dataset, "...")
        gender_acc, age_acc, ethnicity_acc, emotion_acc = evaluate_metrics(predictions, original_labels, available_datasets[args.dataset]["metrics"])
        print("---------------------------------")
        print(args.dataset.upper(), "accuracy:")
        print("Gender", available_datasets[args.dataset]["metrics"]["gender"].__name__, gender_acc) if gender_acc is not None else print("Gender not available")
        print("Age", available_datasets[args.dataset]["metrics"]["age"].__name__, age_acc) if age_acc is not None else print("Age not available")
        print("Ethnicity", available_datasets[args.dataset]["metrics"]["ethnicity"].__name__, ethnicity_acc) if ethnicity_acc is not None else print("Ethnicity not available")
        print("Emotion", available_datasets[args.dataset]["metrics"]["emotion"].__name__, emotion_acc) if emotion_acc is not None else print("Emotion not available")
        print("---------------------------------") 

        print("\nTabulating results in pickle file...")
        summary_results = list()
        if gender_acc is not None:
            summary_results.append(general_summary_data + ["Gender", available_datasets[args.dataset]["metrics"]["gender"].__name__, gender_acc])
        if age_acc is not None:
            summary_results.append(general_summary_data + ["Age", available_datasets[args.dataset]["metrics"]["age"].__name__, age_acc])
        if ethnicity_acc is not None:
            summary_results.append(general_summary_data + ["Ethnicity", available_datasets[args.dataset]["metrics"]["ethnicity"].__name__, ethnicity_acc])
        if emotion_acc is not None:
            summary_results.append(general_summary_data + ["Emotion", available_datasets[args.dataset]["metrics"]["emotion"].__name__, emotion_acc])
        pickle_summary_results = os.path.join(args.outpath, "summary_results.pk")
        with open(pickle_summary_results, "ab") as pkf:
            pickle.dump(summary_results, pkf) 
        print("\nResults pickled with no error.\n")   

    print("Total execution time: %s" % str(datetime.today() - start_time))

    if args.only_inference_time or args.inference_time:
        print("\nRunning inference time test...")
        inference_time, fps, GPU_bytes, params_count = run_inference_time_test(Dataset, args.inpath, partition=args.partition)
        print("\nTabulating inference statistics in pickle file...")
        summary_stats = [general_summary_data + [inference_time, fps, params_count, int(GPU_bytes)]]
        pickle_summary_stats = os.path.join(args.outpath, "summary_stats.pk")
        with open(pickle_summary_stats, "ab") as pkf:
            pickle.dump(summary_stats, pkf) 
        print("\nStatistics pickled with no error.\n\n\n")   
