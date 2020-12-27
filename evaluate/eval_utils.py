import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import sys
sys.path.append("../dataset")
from dataset_utils import raf_age_regression_to_group, fairface_age_regression_to_group


vggface2_raf_race_conversion = {
    0 : 1,
	1 : 2,
	2 : 0,
	3 : 2
}

def categorical_accuracy(predictions, originals):
    return accuracy_score(np.argmax(originals, axis=-1), np.argmax(predictions, axis=-1))

def mae_accuracy(predictions, originals):
    return mean_absolute_error(originals, predictions)

def raf_race_groups_accuracy(predictions, originals):
    converted_predictions = [vggface2_raf_race_conversion[c] for c in np.argmax(predictions, axis=-1)]
    return accuracy_score(np.argmax(originals, axis=-1), converted_predictions)

def raf_age_groups_accuracy(predictions, originals):
    converted_predictions = [raf_age_regression_to_group(f) for f in predictions]
    return accuracy_score(originals, converted_predictions)

def fairface_age_groups_accuracy(predictions, originals):
    converted_predictions = [fairface_age_regression_to_group(f) for f in predictions]
    return accuracy_score(originals, converted_predictions)
