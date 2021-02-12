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

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))

# single task labels
vgg2gender = dict() # dict({id:gender})
vgg2age = dict() # dict({id:age})
vgg2ethnicity = dict() # dict({id:age})
vgg2roi = dict() #

# multitask labels
vgg2partition = dict() # dict({id:partition or None}) # for partitioning purpose
vgg2data = None # dict({image_path: ... }) # for ensembling purpose

available_age_annotation = ["number", "class", "group"]

# ORDER: Gender, Age, Ethnicity, Emotion
def _load_traits(identity_and_roi, gender_meta, age_meta, ethnicity_meta, age_annotation="number"):
    global vgg2data
    if vgg2data is None:
        vgg2data = dict()
        i, errors = 0, defaultdict(set)
        for image_path, image_meta in identity_and_roi.items():
            identity = image_meta["identity"]
            roi = image_meta["roi"]
            try:
                gender = get_gender_meta(gender_meta, identity)
                age = get_age_meta(age_meta, image_path)
                if age_annotation == "class":
                    age =  regression_to_class(age)
                elif age_annotation == "group":
                    age = fairface_age_regression_to_group(age)
                ethnicity = get_ethnicity_meta(ethnicity_meta, identity)
                emotion = MASK_VALUE
                vgg2data[image_path] = {
                    "roi" : roi,
                    "identity" : identity,
                    "gender" : gender,
                    "age" : age,
                    "ethnicity": ethnicity,
                    "emotion": emotion,
                    "sample_num" : i
                }
                i += 1
            except MissingGenderException:
                errors["gender"].add(identity)
            except MissingAgeException:
                errors["age"].add(identity)
            except MissingEthnicityException:
                errors["ethnicity"].add(identity)   
        print("Metadata:", len(vgg2data))
        if errors:
            print("Missing gender:", errors["gender"] if errors["gender"] else "None")
            print("Missing age:", errors["age"] if errors["age"] else "None")
            print("Missing ethnicity:", errors["ethnicity"] if errors["ethnicity"] else "None")


# migliora
def get_gender_meta(gender_meta, identity):
    try:
        gender = gender_meta["n"+identity]
        return get_gender_label(gender)
    except KeyError:
        raise MissingGenderException

def get_age_meta(age_meta, imagepath):
    try:
        age = age_meta[imagepath]
        return get_age_label(age)
    except KeyError:
        raise MissingAgeException

def get_ethnicity_meta(ethnicity_meta, identity):
    try:
        ethnicity = ethnicity_meta["n"+identity]
        return get_ethnicity_label(ethnicity)
    except KeyError:
        raise MissingEthnicityException

# Labelling
def get_gender_label(gender_letter):
    if gender_letter == 'm':
        return LABELS["gender"]["male"]
    elif gender_letter == 'f':
        return LABELS["gender"]["female"]
    raise LabelGenderError


def get_age_label(floating_string, precision=3):
    return float(floating_string) if precision is None else np.round(float(floating_string), precision)


def get_ethnicity_label(ethnicity_index):
    # VMER labels in range 1-4
    return int(ethnicity_index)-1


# Load from csv
def _load_meta_from_csv(csv_meta, output_dict):
    data = readcsv(csv_meta)
    for row in data:
        output_dict[row[0]] = row[1]

def _load_roi_from_csv(csv_meta, output_dict):
    data = readcsv(csv_meta)
    for row in data:
        path =  row[2]
        identity = row[3]
        roi = [int(x) for x in row[4:8]]
        output_dict[path] = {"identity" : identity, "roi" : roi}

def get_partition(identity_label):    
    global vgg2partition
    try:
        faces, partition = vgg2partition[identity_label]
        vgg2partition[identity_label] = (faces + 1, partition)
    except KeyError:
        # split 20/80 stratified by identity
        l = (len(vgg2partition) - 1) % 10
        if l == 0 or l == 1:
            partition = PARTITION_VAL
        else:
            partition = PARTITION_TRAIN
        vgg2partition[identity_label] = (1, partition)
    return partition


def _load_dataset(imagesdir, partition_label, debug_max_num_samples=None):
    data = list()
    discarded_items = defaultdict(list)

    for image_path, image_meta in tqdm(vgg2data.items()):
        path = os.path.join(imagesdir, image_path)
        identity = image_meta["identity"]
        roi = image_meta["roi"]
        roi = enclosing_square(roi)
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
        gender = vgg2data[image_path]["gender"]
        age = vgg2data[image_path]["age"]
        ethnicity = vgg2data[image_path]["ethnicity"]
        emotion = vgg2data[image_path]["emotion"]
        labels = (gender, age, ethnicity, emotion) 
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


class Vgg2DatasetMulti:
    def __init__(self,
                partition='train',
                imagesdir='data/vggface2_data/{part}',
                csvmeta='data/vggface2_data/annotations_multitask/{part}.{task}_{dataset}.csv',
                target_shape=(112, 112, 3),
                augment=True,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None,
                age_annotation="number",
                **kwargs):
                
        partition_label = partition_select(partition)
        if age_annotation not in available_age_annotation:
            raise Exception("Age annotation {} not avalable. Select from {}".format(age_annotation, available_age_annotation))

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        # csv_meta_partition = csvmeta.replace('/', '_').replace('<part>', 'part')
        num_samples = "_"+str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        if age_annotation == "class":
            cache_age = "classedage_"
        elif age_annotation == "group":
            cache_age = "groupedage_"
        else:
            cache_age = ""
        cache_file_name = 'vggface2_multi_{age}{partition}{num_samples}.cache'.format(age=cache_age, partition=partition, num_samples=num_samples)
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

            imagesdir = os.path.join(EXT_ROOT, imagesdir.format(part=load_partition))
            csvmeta = os.path.join(EXT_ROOT, csvmeta)

            roi_csv_meta = csvmeta.format(part=load_partition, task="roi", dataset="detected")
            
            _load_roi_from_csv(roi_csv_meta, vgg2roi)

            # identity_csv_meta = os.path.join(os.path.split(csvmeta)[0], "identity_meta.csv")
            # # global vgg2identity
            # _load_meta_from_csv(identity_csv_meta, vgg2identity)

            gender_csv_meta = csvmeta.format(part=load_partition, task="gender", dataset="vggface2")
            _load_meta_from_csv(gender_csv_meta, vgg2gender)

            age_csv_meta = csvmeta.format(part=load_partition, task="age", dataset="vmage")
            _load_meta_from_csv(age_csv_meta, vgg2age)

            ethnicity_csv_meta = csvmeta.format(part=load_partition, task="ethnicity", dataset="vmer")
            _load_meta_from_csv(ethnicity_csv_meta, vgg2ethnicity)

            _load_traits(vgg2roi, vgg2gender, vgg2age, vgg2ethnicity, age_annotation)
            
            print("Loading {} dataset".format(partition))
            loaded_data = _load_dataset(imagesdir, partition_label, debug_max_num_samples)

            print_verbose_partition(dataset_partition=vgg2partition, verbosed_partition=partition_label)
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

    def get_generator(self, batch_size=64, fullinfo=False, doublelabel=False):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing, 
                                     fullinfo=fullinfo, doublelabel=doublelabel)
        return self.gen

    def get_num_classes(self):
        return CLASSES


def test_multi(dataset="test", debug_samples=None):
    print(dataset, debug_samples if debug_samples is not None else '')
    dt = Vgg2DatasetMulti(dataset,
                        target_shape=(112, 112, 3),
                        preprocessing='vggface2',
                        debug_max_num_samples=debug_samples,
                        age_annotation="class")
    print("SAMPLES %d" % dt.get_num_samples())
    gen = dt.get_generator()

    i = 0
    while True:
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
    # test_multi("train")
    # test_multi("val")
    test_multi("test")
