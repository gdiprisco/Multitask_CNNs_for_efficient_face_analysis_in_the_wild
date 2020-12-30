import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from dataset_VGGFace2_VMER_VMAGE import Vgg2DatasetMulti
from dataset_RAF import RAFDBMulti
from dataset_utils import CLASSES
from dataset_LFWplus import LFWPlusMulti
from dataset_FairFace import FairFaceMulti

import sys
sys.path.append("../training")
from dataset_tools import DataGenerator

from tqdm import tqdm
from cv2 import cv2
import numpy as np


class VGG2RAFDataset:
    def __init__(self, partition, target_shape=(112,112,3), augment=True, custom_augmentation=None, preprocessing='full_normalization', doublelabel=False):
        VGGFace2 = Vgg2DatasetMulti(partition=partition,
                                    target_shape=target_shape,
                                    augment=augment,
                                    custom_augmentation=custom_augmentation,
                                    preprocessing=preprocessing,
                                    age_annotation="number").get_data()
        RAFDB = RAFDBMulti(partition=partition,
                            target_shape=target_shape,
                            augment=augment,
                            custom_augmentation=custom_augmentation,
                            preprocessing=preprocessing,
                            include_gender=True).get_data()
        self.target_shape = target_shape
        self.augment = augment
        self.custom_augmentation = custom_augmentation
        self.preprocessing = preprocessing
        self.rebalance_factor = 3
        # # TRAIN AND VAL/TEST SEPARATE PROTOCOL
        # if partition == "train":
        #     self.data = [VGGFace2, RAFDB]
        #     self.rebalance = True
        # else:
        #     self.data = VGGFace2 + RAFDB
        #     np.random.shuffle(self.data)
        #     self.rebalance = False

        # TRAIN/VAL/TEST SAME PROTOCOL
        self.data = [VGGFace2, RAFDB]
        self.rebalance = True
        
    @staticmethod
    def get_num_classes():
        return CLASSES

    def get_generator(self, batch_size, doublelabel=False):
        return DataGenerator(self.data, target_shape=self.target_shape, with_augmentation=self.augment,
                            custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                            num_classes=CLASSES, preprocessing=self.preprocessing,
                            rebalance=self.rebalance, rebalance_factor=self.rebalance_factor,
                            doublelabel=doublelabel)

    
available_datasets = {
    "VGGFace2-RAF": VGG2RAFDataset,
    "VGGFace2": Vgg2DatasetMulti,
    "RAF": RAFDBMulti,
    "LFW+" : LFWPlusMulti,
    "FairFace" : FairFaceMulti
    }


def load_dataset(dataset):
    if dataset not in available_datasets:
        raise Exception("Dataset {} not available.\nOnly {}".format(dataset, list(available_datasets.keys())))
    else:
        return available_datasets[dataset]

def test_dataset(dataset="VGGFace2-RAF", partition="test", target_shape=(112,112,3), augment=True, custom_augmentation=None, preprocessing="vggface2"):
    print(dataset)
    dataset = VGG2RAFDataset(partition, target_shape, augment, custom_augmentation, preprocessing)
    gen = dataset.get_generator(batch_size=32)
    i = 0
    for batch in tqdm(gen):
        for im, gender, age, ethnicity, emotion in zip(batch[0], batch[1][0], batch[1][1], batch[1][2], batch[1][3]):
            facemax = np.max(im)
            facemin = np.min(im)
            print("Sample:", i)
            print("Labels:", gender, age, ethnicity, emotion)
            im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
            cv2.putText(im, "{} {} {} {}".format(gender, age, ethnicity, emotion), (0, im.shape[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
            cv2.imshow("{} {} {} {}".format(gender, age, ethnicity, emotion), im)
            i += 1
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    test_dataset(partition="test", augment=False)
