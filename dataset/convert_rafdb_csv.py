import os
from collections import defaultdict
import csv
from tqdm import tqdm

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
emo_list_partition_label = os.path.join(EXT_ROOT, "data/RAF-DB/basic/EmoLabel/list_patition_label.txt")
annotation_file = os.path.join(EXT_ROOT, "data/RAF-DB/basic/Annotation/manual/{sample_name}_manu_attri.txt")
output_file = os.path.join(EXT_ROOT, "data/RAF-DB/basic/multitask/{part}.multitask_rafdb.csv")

labels = {
    "gender" : {
        0: "male",
        1: "female",
        2: "unsure"
    },
    "age_group": {
        0: "0-3",
        1: "4-19",
        2: "20-39",
        3: "40-69",
        4: "70+"
    },
    "race": {
        0: "Caucasian",
        1: "African-American",
        2: "Asian"
    },
    "emotion": {
        1: "Surprise",
        2: "Fear",
        3: "Disgust",
        4: "Happiness",
        5: "Sadness",
        6: "Anger",
        7: "Neutral"
    }
}

data = defaultdict(dict)

def extract_emo(emo_txt):
    with open(emo_txt) as fp:
        for line in fp.readlines():
            tmp_line = line.splitlines()[0].split(" ")
            data[tmp_line[0]]["emotion"] = labels["emotion"][int(tmp_line[1])]

def extract_gender_race_agegroup(annotation_regex):
    data_keys = list(data.keys())
    for sample_name in data_keys:
        splitted_sample_name = os.path.splitext(sample_name)[0]
        sample_annotation_file = annotation_regex.format(sample_name=splitted_sample_name)
        with open(sample_annotation_file) as fp:
            sample_meta = fp.read().splitlines()
        gender_meta = int(sample_meta[5])
        data[sample_name]["gender"] = labels["gender"][gender_meta]
        race_meta = int(sample_meta[6])
        data[sample_name]["race"] = labels["race"][race_meta]
        age_group = int(sample_meta[7])
        data[sample_name]["age_group"] = labels["age_group"][age_group]
        

def write_csv(output_file):
    with open(output_file.format(part="train"), "w") as fp:
        print("Writing train data")
        train_writer = csv.writer(fp, delimiter=",")
        for sample_name, sample_meta in tqdm(data.items()):
            if sample_name.startswith("train"):
                meta = [sample_name, sample_meta["gender"], sample_meta["age_group"], sample_meta["race"], sample_meta["emotion"]]
                train_writer.writerow(meta)
    with open(output_file.format(part="test"), "w") as fp:
        print("Writing test data")
        test_writer = csv.writer(fp, delimiter=",")
        for sample_name, sample_meta in tqdm(data.items()):
            if sample_name.startswith("test"):
                meta = [sample_name, sample_meta["gender"], sample_meta["age_group"], sample_meta["race"], sample_meta["emotion"]]
                test_writer.writerow(meta)


if __name__ == '__main__':
    extract_emo(emo_list_partition_label)
    extract_gender_race_agegroup(annotation_file)
    write_csv(output_file)

