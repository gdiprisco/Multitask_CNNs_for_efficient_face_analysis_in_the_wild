import csv
import numpy as np
import unicodedata

def _readcsv(csvpath, debug_max_num_samples=None):
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

def read():
    # FIXME external inpath 
    inpath = '/user/gdiprisco/multitask/dataset/data/vggface2_data/annotations/identity_meta.csv'

    train = list()
    test = list()

    identities = _readcsv(inpath)

    for row in identities[1:]:
        if len(row) == 5:
            if row[3] == '0':           
                test.append(row[0] + "," + row[1])
            elif row[3] == '1':
                train.append(row[0] + "," + row[1])
            else:
                print(row[0])
        elif len(row) == 6:
            if row[4] == '0':
                test.append(row[0] + "," + row[1] + "," + row[2])
            elif row[4] == '1':
                train.append(row[0] + "," + row[1] + "," + row[2])
            else:
                print(row[0])
        else:
            print("len", row[0], len(row))

    return train, test


def write(train, test):
    with open('/user/gdiprisco/multitask/dataset/data/vggface2_data/annotations/train.identity_vggface2.csv', 'a') as fp:
        for row in train:
            text = unicodedata.normalize('NFKD', row).encode('ascii', 'ignore')
            fp.write(str(text))
    with open('/user/gdiprisco/multitask/dataset/data/vggface2_data/annotations/test.identity_vggface2.csv', 'a') as fp:
        for row in test:
            text = unicodedata.normalize('NFKD', row).encode('ascii', 'ignore')
            fp.write(str(text))