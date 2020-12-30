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
# sys.path.append("../training/scratch_models")
# sys.path.append('../training/keras_vggface/keras_vggface')

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
parser.add_argument('--checkpoint', dest="checkpoint", type=int, help="Specify checkpoint if 'path' is the main directory of the model")
args = parser.parse_args()

custom_objects = {
                #   'BlurPool': BlurPool,
                #   'relu6': relu6,
                'age_relu': age_relu,
                'Hswish': Hswish,
                'HSigmoid': HSigmoid
}


def load_keras_model(filepath):
    version = re.search("versionver[ABC]", os.path.split(os.path.split(filepath)[0])[1])
    if not version:
        raise Exception("Unable to infer model version from path splitting")
    version = version[0].replace("version", "")
    loss, loss_weights, accuracy_metrics, _ = get_versioned_metrics(version)
    model = keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
    model.compile(loss=loss, loss_weights=loss_weights, optimizer='sgd', metrics=accuracy_metrics)
    INPUT_SHAPE = (112, 112, 3)
    return model, INPUT_SHAPE

def get_filepath_ck(main_filepath, checkpoint):
    tail = 'checkpoint.{}.hdf5'.format(str(checkpoint).zfill(2))
    return os.path.join(main_filepath, tail)

# def evalds(Dataset, filepath, outf_path, partition, batch_size=64):
#     print('Partition: %s' % partition)
#     outf = open(outf_path, "a+")
#     outf.write('Results for: %s\n' % filepath)
#     model, INPUT_SHAPE = load_keras_model(filepath)

#     dataset_test = Dataset(partition, target_shape=INPUT_SHAPE, augment=False, preprocessing='vggface2')

#     data_gen = dataset_test.get_generator(batch_size)
#     print("Dataset batches %d" % len(data_gen))
#     start_time = time.time()
#     result = model.evaluate_generator(data_gen, verbose=1, workers=4)
#     spent_time = time.time() - start_time
#     batch_average_time = spent_time / len(data_gen)
#     print("Evaluate time %d s" % spent_time)
#     print("Batch time %.10f s" % batch_average_time)
#     o = "%s %f\n" % (partition, result[1])
#     print("\n\n RES " + o)
#     outf.write(o)

#     outf.write('\n\n')
#     outf.close()

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
    print("Evaluate time %d s" % spent_time)
    print("Batch time %.10f s, FPS: %.3f" % (batch_average_time, 1/batch_average_time))
    GPU_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=1)
    print(" --- INFERENCE TEST RUNNED ---")
    print("Memory usage {} bytes".format(GPU_bytes))
    print(" -----------------------------")


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
    # gender_acc = metric(predictions["gender"], originals["gender"]) if metric is not None else None
    # age_acc = metric(predictions["age"], originals["age"]) if metric is not None else None
    # ethnicity_acc = metric(predictions["ethnicity"], originals["ethnicity"]) if metric is not None else None
    # emotion_acc = metric(predictions["emotion"], originals["emotion"]) if metric is not None else None
    # return gender_acc, age_acc, ethnicity_acc, emotion_acc
    

#TODO add inference (single sample) mode

if '__main__' == __name__:
    if not args.inpath.endswith('.hdf5'): raise Exception("Only .hdf5 files are supported.")
    start_time = datetime.today()
    os.makedirs(args.outpath, exist_ok=True)
    Dataset = available_datasets[args.dataset]["dataset"]

    # results_filename = "{}_{}_results".format(args.dataset, args.partition)

    # if args.time:
    #     out_path = os.path.join(args.outf, "{}_{}.txt".format(results_filename, start_time.strftime('%Y%m%d_%H%M%S')))
    # else:
    #     out_path = os.path.join(args.outf, results_filename+".txt")
    
    if args.gpu is not None:
        gpu_to_use = [str(s) for s in args.gpu.split(',') if s.isdigit()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    # if args.inpath.endswith('.hdf5'):
    #     in_path = args.inpath,
    #     out_path = out_path
    # elif args.checkpoint is not None:
    #     in_path = get_filepath_ck(args.inpath, args.checkpoint),
    #     out_path = "{}_{}{}".format(os.path.splitext(out_path)[0], "checkpoint_{}".format(args.checkpoint), os.path.splitext(out_path)[1])
    # else:
    #     raise Exception("Only .hdf5 files are supported.")
    
    model_dirname, checkpoint = os.path.split(os.path.split(args.inpath)[0])[1], os.path.split(args.inpath)[1].split(".")[1]
    # out_path = "results_{}_{}_of_checkpoint_{}_from{}.csv".format(args.dataset, args.partition, checkpoint, model_dirname)
    out_path = "results_{}_{}_of_checkpoint_{}_from{}".format(args.dataset, args.partition, checkpoint, model_dirname)
    out_path = os.path.join(args.outpath, out_path)
    pk_out_path = out_path+".pk"
    csv_out_path = out_path+".csv"
    # if os.path.exists(pk_out_path):
    #     print("LOADING CACHED RESULTS...")
    #     # reference = _refactor_data(readcsv(out_path)) # TODO
    #     pk_data = pickle.load(open(pk_out_path, "rb"))
    #     # image_paths = pk_data["image_paths"]
    #     predictions = pk_data["predictions"]
    #     original_labels = pk_data["original_labels"]
    #     # image_rois = pk_data["image_rois"]
    # else:
    
    print("Running test from scratch ...")
    image_paths, predictions, original_labels, image_rois  = run_test(Dataset, args.inpath, batch_size=64, partition=args.partition)
    
    # print(image_paths[0])
    # print(predictions["gender"][0])
    # print(original_labels["gender"][0])
    # print(predictions["age"][0])
    # print(original_labels["age"][0])
    # print(predictions["ethnicity"][0])
    # print(original_labels["ethnicity"][0])
    # print(predictions["emotion"][0])
    # print(original_labels["emotion"][0])
    # print(image_rois[0])
    # exit(1)
    print("Managing data ...")
    reference = zip_reference(image_paths, predictions, original_labels, image_rois)
    # for i in reference[0]:
    #     print(i)
    # exit()
    print("Writing CSV", csv_out_path, "...")
    writecsv(csv_out_path, reference) # TODO better format CSV
    print("Writing Pickle", pk_out_path, "...")
    pk_data = {
        "image_paths" : image_paths,
        "predictions" : predictions,
        "original_labels" : original_labels,
        "image_rois" : image_rois
    }
    pickle.dump(pk_data, open(pk_out_path, "wb"))

    print("Evaluating metrics for", args.dataset, "...")
    gender_acc, age_acc, ethnicity_acc, emotion_acc = evaluate_metrics(predictions, original_labels, available_datasets[args.dataset]["metrics"])
    print("---------------------------------")
    print(args.dataset.upper(), "accuracy:")
    print("Gender", available_datasets[args.dataset]["metrics"]["gender"].__name__, gender_acc) if gender_acc is not None else print("Gender not available")
    print("Age", available_datasets[args.dataset]["metrics"]["age"].__name__, age_acc) if age_acc is not None else print("Age not available")
    print("Ethnicity", available_datasets[args.dataset]["metrics"]["ethnicity"].__name__, ethnicity_acc) if ethnicity_acc is not None else print("Ethnicity not available")
    print("Emotion", available_datasets[args.dataset]["metrics"]["emotion"].__name__, emotion_acc) if emotion_acc is not None else print("Emotion not available")
    print("---------------------------------") 

    # print("#################################")
    # model, INPUT_SHAPE = load_keras_model(args.inpath)
    # loss = {
    #     "gen1" : build_masked_loss(keras.losses.binary_crossentropy),
    #     # "age1" : build_masked_loss(keras.losses.binary_crossentropy),
    #     "age1" : build_masked_loss(keras.losses.mean_squared_error),
    #     "eth1" : build_masked_loss(keras.losses.binary_crossentropy),
    #     "emo1" : build_masked_loss(keras.losses.binary_crossentropy)
    # }

    # loss_weights = {
    #     "gen1" : 1.0,
    #     # "age1" : 1.0,
    #     "age1" : 0.025,
    #     "eth1" : 1.0,
    #     "emo1" : 1.0
    # }

    # accuracy = {
    #     "gen1" : build_masked_acc(keras.metrics.categorical_accuracy),
    #     # "age1" : build_masked_acc(keras.metrics.categorical_accuracy),
    #     "age1" : build_masked_acc(keras.metrics.mean_absolute_error),
    #     "eth1" : build_masked_acc(keras.metrics.categorical_accuracy),
    #     "emo1" : build_masked_acc(keras.metrics.categorical_accuracy)
    # }
    # model.compile(loss=loss, loss_weights=loss_weights, optimizer="sgd", metrics=accuracy)
    # dataset = Dataset(partition=args.partition,
    #                   target_shape=(112,112,3),
    #                   augment=False,
    #                   preprocessing='vggface2',
    #                   age_annotation="number",
    #                   include_gender=True,
    #                   include_age_group=True,
    #                   include_race=True)


    print("Total execution time: %s" % str(datetime.today() - start_time))

    print("\nRunning inference time test...")
    run_inference_time_test(Dataset, args.inpath, partition=args.partition)

    