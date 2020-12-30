from abc import ABC, abstractmethod
import keras

def age_relu(x):
    return keras.backend.relu(x, max_value=100)

STANDARD_CLASSES = {
    "gender": 2,
    # "age": 5, 
    "age": 1,
    "ethnicity": 4,
    "emotion": 7
}

STANDARD_ACT = {
    "gender": "softmax",
    # "age": "softmax",
    "age": age_relu,
    "ethnicity": "softmax",
    "emotion": "softmax"
}

class Model(ABC):

    __slots__ = ["input_shape", "classes", "weights", "model", "_joint_bottom", "_disjoint_bottom"]

    def __init__(self, input_shape=(112, 112, 3), weights=None):
        self.input_shape = input_shape
        self.model = None
        self._joint_bottom = None
        self._disjoint_bottom = None
        
        self.buildable = True
        self._available_joint_branches = ["gen1", "age1", "eth1", "emo1", "gen2", "age2", "eth2", "emo2"]
        self._available_disjoint_branches = ["genb", "ageb", "ethb", "emob"]

    def baseline(self, num_classes=2, activation="softmax"):
        return keras.models.Model(self.model.input, self._joint_top(self._joint_bottom, num_classes, activation))

    def check_buildable(self):
        if not self.buildable:
            raise Exception("Unbuildable network.\
                            Maybe you called two methods on the Model object.\
                            You can obtain one version of a model from a single Model object.\
                            Please create new Model and re-call your method.")

    def set_unbuildable(self):
        self.buildable = False

    @abstractmethod
    def _joint_top(self, features, num_classes, activation): 
        pass

    def _joint_branches(self, features): 
        gender = self._joint_top(features, STANDARD_CLASSES["gender"], STANDARD_ACT["gender"])
        age = self._joint_top(features, STANDARD_CLASSES["age"], STANDARD_ACT["age"])
        ethnicity = self._joint_top(features, STANDARD_CLASSES["ethnicity"], STANDARD_ACT["ethnicity"])
        emotion = self._joint_top(features, STANDARD_CLASSES["emotion"], STANDARD_ACT["emotion"])
        return gender, age, ethnicity, emotion

    def joint_extraction_model(self): 
        self.check_buildable()
        gender, age, ethnicity, emotion = self._joint_branches(features=self._joint_bottom)
        joint_model = keras.models.Model(self.model.input, [gender, age, ethnicity, emotion])
        self.set_unbuildable()
        return joint_model

    @abstractmethod
    def _disjoint_top(self, features, num_classes, activation, improved):
        pass

    def _disjoint_branches(self, improved=False): 
        gender = self._disjoint_top(self._disjoint_bottom, STANDARD_CLASSES["gender"], STANDARD_ACT["gender"], improved)
        age = self._disjoint_top(self._disjoint_bottom, STANDARD_CLASSES["age"], STANDARD_ACT["age"], improved)
        ethnicity = self._disjoint_top(self._disjoint_bottom, STANDARD_CLASSES["ethnicity"], STANDARD_ACT["ethnicity"], improved)
        emotion = self._disjoint_top(self._disjoint_bottom, STANDARD_CLASSES["emotion"], STANDARD_ACT["emotion"], improved)
        return gender, age, ethnicity, emotion

    def disjoint_extraction_model(self):
        self.check_buildable()
        gender, age, ethnicity, emotion = self._disjoint_branches()
        disjoint_model = keras.models.Model(self.model.input, [gender, age, ethnicity, emotion])
        self.set_unbuildable()
        return disjoint_model

    @abstractmethod
    def _aggregate_low_level_features(self, features):
        pass

    def improved_disjoint_extraction_model(self): 
        self.check_buildable()
        gender1, age1, ethnicity1, emotion1 = self._disjoint_branches(improved=True)
        low_level_features = self._aggregate_low_level_features(self._disjoint_bottom)
        concatenation = keras.layers.Concatenate()([low_level_features, gender1, age1, ethnicity1, emotion1])
        gender2, age2, ethnicity2, emotion2 = self._joint_branches(features=concatenation)
        # improved_model = keras.models.Model(self.model.input, [gender2, age2, ethnicity2, emotion2])
        improved_model = keras.models.Model(self.model.input, [gender1, age1, ethnicity1, emotion1, gender2, age2, ethnicity2, emotion2])
        self.set_unbuildable()
        return improved_model