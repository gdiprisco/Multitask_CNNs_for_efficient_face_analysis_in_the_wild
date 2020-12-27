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

def FF_VMER_conversion(FF_label):
    switcher = {
        "Black" : "African American",
        "Indian" : "Asian Indian",
        "East Asian" : "East Asian",
        "Southeast Asian" : "East Asian",
        "Middle Eastern" : "Caucasian Latin", 
        "White" : "Caucasian Latin", 
        "Latino_Hispanic" : "Caucasian Latin"
    }
    return switcher[FF_label]

fairface_age_labels = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6, 
    "60-69": 7,
    "70": 8,
    "more than 70": 9
}

def _readcsv(csvpath, debug_max_num_samples=None):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


def _load_FairFace_multi(csvmeta, imagesdir, debug_max_num_samples=None):
    meta = _readcsv(csvmeta, debug_max_num_samples)[1:]
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    data = []
    n_discarded = 0
    for d in tqdm(meta):
        # service_test = eval(d[4])
        gender = LABELS["gender"][d[2].lower()]
        age = fairface_age_labels[d[1]]
        ethnicity = LABELS["ethnicity"][FF_VMER_conversion(d[3])]
        emotion = MASK_VALUE
        path = os.path.join(imagesdir, d[0])
        img = cv2.imread(path)
        if img is not None:
            roi = [0, 0, img.shape[1], img.shape[0]]
            example = {
                'img': path,
                'label': (gender, age, ethnicity, emotion),
                'roi': roi,
                'part': PARTITION_TEST
            }
            if np.max(img) == np.min(img):
                print('Warning, blank image: %s!' % path)
            else:
                data.append(example)
        else:  # img is None
            print("WARNING! Unable to read %s" % path)
            n_discarded += 1
    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    return data


class FairFaceMulti:
    def __init__(self, partition='test',
                imagesdir='data/fairface',
                csvmeta='data/fairface/fairface_label_val.csv',
                target_shape=(112, 112, 3),
                augment=False,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None,
                **kwargs):
                
        if not partition.startswith('test'):
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        print('Loading %s data...' % partition)

        num_samples = "_" + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'fairface_gender_agegroups_ethnicity_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)
        
        cache_file_name = os.path.join("cache", cache_file_name)
        cache_file_name = os.path.join(EXT_ROOT, cache_file_name)

        print("cache file name %s" % cache_file_name)
        try:
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except FileNotFoundError:
            print("Loading %s data from scratch..." % partition)
            csvmeta = os.path.join(EXT_ROOT, csvmeta)
            imagesdir = os.path.join(EXT_ROOT, imagesdir)
            self.data = _load_FairFace_multi(csvmeta, imagesdir, debug_max_num_samples)
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping...")
                pickle.dump(self.data, f)

    def get_generator(self, batch_size=64, fullinfo=False):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing,
                                     fullinfo=fullinfo)
        return self.gen

    def get_num_classes(self):
        return CLASSES

    def get_num_samples(self):
        return len(self.data)

    def get_data(self):
        return self.data


def test_multi(partition="test", debug_samples=None):
    print("Partion", partition, debug_samples if debug_samples is not None else '')
    dataset = FairFaceMulti(partition=partition,
                            target_shape=(112, 112, 3),
                            preprocessing='vggface2',
                            augment=False,
                            debug_max_num_samples=debug_samples)
    print("Samples in dataset partition", dataset.get_num_samples())

    gen = dataset.get_generator(fullinfo=True)
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

if __name__ == "__main__":
    test_multi()