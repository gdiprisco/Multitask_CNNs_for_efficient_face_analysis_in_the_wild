import numpy as np
import csv

MASK_VALUE = -1
MASK_VERBOSE = "Und"
CLASSES = (2, 1, 4, 7)
LABELS = {
    "gender" : {
        "female" : 0,
        "male" : 1
    },
    "age_group" : {
        "0-3" : 0,
        "4-19" : 1,
        "20-39" : 2,
        "40-69" : 3,
        "70+" : 4 
    },
    # "ethnicity_rev" : {
    #     0: "African American",
	#     1: "East Asian",
	#     2: "Caucasian Latin",
	#     3: "Asian Indian"
    # },
    "ethnicity" : {
        "African American" : 0,
	    "East Asian":1,
	    "Caucasian Latin" : 2,
	    "Asian Indian" : 3
    },
    "emotion":{
        "Surprise": 0,
        "Fear": 1,
        "Disgust": 2,
        "Happiness" : 3,
        "Sadness": 4,
        "Anger": 5,
        "Neutral": 6
    }
}

PARTITION_TRAIN = 0
PARTITION_VAL = 1
PARTITION_TEST = 2

class MissingGenderException(Exception):
    pass

class MissingAgeException(Exception):
    pass

class MissingEthnicityException(Exception):
    pass

class LabelGenderError(Exception):
    pass

class LabelAgeError(Exception):
    pass

class LabelEthnicityError(Exception):
    pass

def readcsv(csvpath, debug_max_num_samples=None):
    data = list()
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True,
                            delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


def writecsv(path, data):
    with open(path,"w") as write_obj:
        writer = csv.writer(write_obj)
        for line in data:
            if line[3] is not None:
                extended_line = line[:3] + tuple(line[3])
                writer.writerow(extended_line)
            else:
                writer.writerow(line)

def partition_select(partition):
    if partition.startswith('train'):
        return PARTITION_TRAIN
    elif partition.startswith('val'):
        return PARTITION_VAL
    elif partition.startswith('test'):
        return PARTITION_TEST
    else:
        raise Exception("unknown partition")


def print_verbose_partition(dataset_partition, verbosed_partition):
    if verbosed_partition == PARTITION_TRAIN or verbosed_partition == PARTITION_VAL:
        train_identities, train_samples = 0, 0
        val_identities, val_samples = 0, 0

        print("Verbose partitions...")
        
        for _, (faces, partition) in dataset_partition.items():
            if partition == PARTITION_TRAIN:
                train_samples += faces
                train_identities += 1
            elif partition == PARTITION_VAL:
                val_samples += faces
                val_identities += 1

        train_identities_percentage = 100 * train_identities / (train_identities + val_identities)
        train_samples_percentage = 100 * train_samples / (train_samples + val_samples)

        print("Train identities {} ({}% of all identites)".format(train_identities, train_identities_percentage))
        print("Train samples {} ({}% of all identites)".format(train_samples, train_samples_percentage))

        val_identities_percentage = 100 * val_identities / (train_identities + val_identities)
        val_samples_percentage = 100 * val_samples / (train_samples + val_samples)

        print("validation identities {} ({}% of all identites)".format(val_identities, val_identities_percentage))
        print("Validation samples {} ({}% of all identites)".format(val_samples, val_samples_percentage))

def _reverse_labels(subkey, value):
    for k,v in LABELS[subkey].items():
        if v == value:
            return k
    return None

def _verbose_label(task, hot_code):
    return _reverse_labels(task, np.argmax(hot_code)) if np.sum(hot_code)>0 else MASK_VERBOSE

def verbose_gender(hot_code):
    return _verbose_label("gender", hot_code)

def verbose_age(age):
    return MASK_VERBOSE if age == MASK_VALUE else age

def verbose_ethnicity(hot_code):
    return _verbose_label("ethnicity", hot_code)

def verbose_emotion(hot_code):
    return _verbose_label("emotion", hot_code)

def raf_age_regression_to_group(age_float):
    if round(age_float) < 4:
        return 0
    elif round(age_float) < 19:
        return 1
    elif round(age_float) < 39:
        return 2
    elif round(age_float) < 69:
        return 3
    else:
        return 4

def fairface_age_regression_to_group(age_float):
    # "0-2": 0,
    # "3-9": 1,
    # "10-19": 2,
    # "20-29": 3,
    # "30-39": 4,
    # "40-49": 5,
    # "50-59": 6, 
    # "60-69": 7,
    # "70": 8,
    # "more than 70": 9

    if round(age_float) < 3:
        return 0
    elif round(age_float) > 70:
        return 9
    else:
        return round(age_float/10)+1

def regression_to_class(age_float):
    # 101 classes - from 0 to 100
    age_rounded = int(round(age_float))
    return age_rounded if age_rounded <= 100 else 100
        
    
    