import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys
from collections import defaultdict

from dataset_utils import *

sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin, DataGenerator

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

rafdb_labels = {
    "age_group": {
        "0-3": 0,
        "4-19": 1,
        "20-39": 2,
        "40-69": 3,
        "70+":4 
    },
    "race": {
        "Caucasian": 0,
        "African-American": 1,
        "Asian": 2
    }
}

# converted labels
rafDBmeta = defaultdict(dict)

# multitask labels
rafDBpartition = dict() # dict({id:partition or None}) # for partitioning purpose
rafDBdata = None # dict({image_path: ... }) # for ensembling purpose


# ORDER: Gender, Age, Ethnicity, Emotion
def _load_traits(input_meta, include_gender=False, include_age_group=False, include_race=False):
    global rafDBdata
    if rafDBdata is None:
        rafDBdata = dict()
        i, errors = 0, defaultdict(set)
        for image_path, image_meta in input_meta.items():
            identity = image_meta["identity"]
            roi = None # aligned image, roi is the image size
            rafDBdata[image_path] = {
                "roi" : roi,
                "identity" : identity,
                "gender" : get_gender_label(image_meta["gender"]) if include_gender else MASK_VALUE,
                "age_group" : get_age_group_label(image_meta["age_group"]) if include_age_group else MASK_VALUE,
                "ethnicity": get_ethnicity_label(image_meta["race"]) if include_race else MASK_VALUE,
                "emotion": get_emotion_label(image_meta["emotion"]),
                "sample_num" : i
            }
            i += 1   
        print("Metadata:", len(rafDBdata))
        if errors:
            print("Gender errors", errors["gender"])
            print("Age errors", errors["age"])
            print("Ethnicity errors", errors["ethnicity"])


# Labelling
def get_gender_label(gender):
    if gender == 'male':
        return LABELS["gender"]["male"]
    elif gender == 'female':
        return LABELS["gender"]["female"]
    return MASK_VALUE

def get_age_group_label(age_group_text):
    return rafdb_labels["age_group"][age_group_text]

def get_ethnicity_label(ethnicity_text):
    return rafdb_labels["race"][ethnicity_text]

def get_emotion_label(emotion):
    return LABELS["emotion"][emotion]


# Load from csv
def _load_meta_from_csv(csv_meta, output_dict):
    data = readcsv(csv_meta)
    for row in data:
        output_dict[row[0]]["gender"] = row[1]
        output_dict[row[0]]["age_group"] = row[2]
        output_dict[row[0]]["race"] = row[3]
        output_dict[row[0]]["emotion"] = row[4]
        output_dict[row[0]]["identity"] = row[0].split("_")[1]


def get_partition(identity_label):    
    global rafDBpartition
    try:
        faces, partition = rafDBpartition[identity_label]
        rafDBpartition[identity_label] = (faces + 1, partition)
    except KeyError:
        # split 20/80 stratified by identity
        l = (len(rafDBpartition) - 1) % 10
        if l == 0 or l == 1:
            partition = PARTITION_VAL
        else:
            partition = PARTITION_TRAIN
        rafDBpartition[identity_label] = (1, partition)
    return partition


def _load_dataset(imagesdir, partition_label, debug_max_num_samples=None):
    data = list()
    discarded_items = defaultdict(list)

    for image_path, image_meta in tqdm(rafDBdata.items()):
        path = os.path.join(imagesdir, image_path)
        if ALIGNED:
            path = os.path.splitext(path)
            path = path[0] + "_aligned" + path[1]
        identity = image_meta["identity"]
        image = cv2.imread(path)
        if image is None:
            print("WARNING! Unable to read {}".format(image_path))
            print(" - At {}".format(path))
            discarded_items["unavailable_image"].append(identity)
            continue
        if np.max(image) == np.min(image):
            print("Blank image {}".format(image_path))
            discarded_items["blank_image"].append(identity)
            continue
        sample_partition = PARTITION_TEST if partition_label == PARTITION_TEST else get_partition(identity)
        gender = rafDBdata[image_path]["gender"]
        age = rafDBdata[image_path]["age_group"]
        ethnicity = rafDBdata[image_path]["ethnicity"]
        emotion = rafDBdata[image_path]["emotion"]
        labels = (gender, age, ethnicity, emotion)
        roi = (0, 0, image.shape[1], image.shape[0]) if image_meta["roi"] is None else image_meta["roi"] 
        sample = {
            'img': path,
            'label': labels,
            'roi': roi,
            'part': sample_partition
        }
        data.append(sample)
        if debug_max_num_samples is not None and len(data) >= debug_max_num_samples:
            print("Stopped loading. Debug max samples: ", debug_max_num_samples)
            break
    print("Data loaded. {} samples".format(len(data)))
    print("Discarded for unavailable image: ", len(discarded_items["unavailable_image"]))
    print("Discarded for blank image: ", len(discarded_items["blank_image"]))
    return data


ALIGNED = True

class RAFDBMulti:
    def __init__(self,
                partition='train',
                imagesdir='data/RAF-DB/basic/Image/{aligned}',
                csvmeta='data/RAF-DB/basic/multitask/{part}.multitask_rafdb.csv',
                target_shape=(112, 112, 3),
                augment=True,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None,
                include_gender=False,
                include_age_group=False,
                include_race=False,
                **kwargs):
                
        partition_label = partition_select(partition)

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_" + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_task = "{}{}{}_emotion".format(
            "_withgender" if include_gender else "",
            "_withagegroup" if include_age_group else "",
            "_withrace" if include_race else ""
        )
        cache_file_name = 'rafdb{task}_{partition}{num_samples}.cache'.format(task=cache_task, partition=partition, num_samples=num_samples)
        cache_file_name = os.path.join("cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)
        print("cache file name %s" % cache_file_name)

        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch" % partition)
            load_partition = "train" if partition_label == PARTITION_TRAIN or partition_label == PARTITION_VAL else "test"

            imagesdir = os.path.join(EXT_ROOT, imagesdir.format(aligned="aligned" if ALIGNED else "original"))
            csvmeta = os.path.join(EXT_ROOT, csvmeta.format(part=load_partition))

            _load_meta_from_csv(csvmeta, rafDBmeta)

            _load_traits(rafDBmeta, include_gender, include_age_group, include_race)
            
            print("Loading {} dataset".format(partition))
            loaded_data = _load_dataset(imagesdir, partition_label, debug_max_num_samples)

            print_verbose_partition(dataset_partition=rafDBpartition, verbosed_partition=partition_label)
            if partition.startswith('test'):
                self.data = loaded_data
            else:
                self.data = [x for x in loaded_data if x['part'] == partition_label]
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping")
                pickle.dump(self.data, f)

    def get_data(self):
        return self.data

    def get_num_samples(self):
        return len(self.data)

    def get_generator(self, batch_size=64, fullinfo=False):
        if self.gen is None:
            self.gen = DataGenerator(data=self.data,
                                    target_shape=self.target_shape,
                                    with_augmentation=self.augment,
                                    custom_augmentation=self.custom_augmentation,
                                    batch_size=batch_size,
                                    num_classes=self.get_num_classes(),
                                    preprocessing=self.preprocessing, 
                                    fullinfo=fullinfo)
        return self.gen

    def get_num_classes(self):
        return CLASSES


def test_multi(dataset="test", debug_samples=None):

    if dataset.startswith("train") or dataset.startswith("val"):
        print(dataset, debug_samples if debug_samples is not None else '')
        dt = RAFDBMulti(dataset,
                        target_shape=(112, 112, 3),
                        preprocessing='vggface2',
                        debug_max_num_samples=debug_samples)
        gen = dt.get_generator()
    else:
        dv = RAFDBMulti('test',
                        target_shape=(112, 112, 3),
                        preprocessing='vggface2',
                        debug_max_num_samples=debug_samples)
        gen = dv.get_generator()
    i = 0
    for batch in tqdm(gen):
        for im, gender, age, ethnicity, emotion in zip(batch[0], batch[1][0], batch[1][1], batch[1][2], batch[1][3]):
            facemax = np.max(im)
            facemin = np.min(im)
            print("Sample:", i)
            print("Labels:", gender, age, ethnicity, emotion)
            print("Gender:", verbose_gender(gender),
                    "- Age:", verbose_age(age),
                    "- Ethnicity:", verbose_ethnicity(ethnicity),
                    "- Emotion:", verbose_emotion(emotion))
            im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
            cv2.putText(im, "{} {} {} {}".format(verbose_gender(gender), verbose_age(age), verbose_ethnicity(ethnicity), verbose_emotion(emotion)),
                        (0, im.shape[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            cv2.imshow("{} {} {} {}".format(verbose_gender(gender), verbose_age(age), verbose_ethnicity(ethnicity), verbose_emotion(emotion)), im)
            i += 1
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return


if '__main__' == __name__:
    test_multi("train")
    test_multi("val")
    test_multi("test")
