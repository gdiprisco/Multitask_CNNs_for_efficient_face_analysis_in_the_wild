import keras
import os
import tensorflow as tf
from model_abstract import Model, STANDARD_CLASSES, STANDARD_ACT
from keras_applications import resnet_common


class ResNet50V2(Model):
    def __init__(self, input_shape=(112, 112, 3), weights="imagenet"):
        print("Building ResNet50V2", weights if weights is not None else "", input_shape)
        self.input_shape = input_shape
        self.model = keras.applications.ResNet50V2(input_shape=input_shape, weights=weights, include_top=False, pooling="avg")
        self._joint_bottom = self.model.layers[-1].output
        self._disjoint_bottom = self.model.layers[-38].output 
        self.buildable = True
        self._available_joint_branches = ["gen1", "age1", "eth1", "emo1", "gen2", "age2", "eth2", "emo2"]
        self._available_disjoint_branches = ["genb", "ageb", "ethb", "emob"]
    
    def _joint_top(self, features, num_classes, activation):
        current_dense = self._available_joint_branches.pop(0)
        return keras.layers.Dense(num_classes, use_bias=True, activation=activation, name=current_dense)(features)

    def _disjoint_top(self, features, num_classes, activation, improved=False):
        current_branch = self._available_disjoint_branches.pop(0)
        # appending the 4th block
        top_features = resnet_common.stack2(features, 512, 3, stride1=1, name='conv5_'+current_branch)
        # appending feature extraction tails
        top_features = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='post_bn_'+current_branch)(top_features)
        top_features = keras.layers.Activation('relu', name='post_relu_'+current_branch)(top_features)
        # Average Pooling 2D previously removed
        top_features = keras.layers.GlobalAveragePooling2D(name='avg_pool_'+current_branch)(top_features)
        return self._joint_top(top_features, num_classes, activation)

    def _aggregate_low_level_features(self, features):
        # Same solution of the pre-built feature extraction tail
        aggregate = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='post_bn_features')(features)
        aggregate = keras.layers.Activation('relu', name='post_relu_features')(aggregate)
        return keras.layers.GlobalAveragePooling2D(name='avg_pool_features')(aggregate)


def test(text_gpu="0"):
    directory = "test_models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(text_gpu)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    print("Creating modelbase...")
    modelbase = ResNet50V2()
    print("Modelbase created")
    print("Model Baseline")
    model = modelbase.baseline()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/ResNet50V2_baseline.h5".format(directory))
    print("Baseline model saved.")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = ResNet50V2()
    print("Modelbase created")
    print("Model Ver. A")
    model = modelbase.joint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/ResNet50V2_ver_A.h5".format(directory))
    print("Ver. A model saved.")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = ResNet50V2()
    print("Modelbase created")
    print("Model Ver. B")
    model = modelbase.disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/ResNet50V2_ver_B.h5".format(directory))
    print("Ver. B model saved.")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = ResNet50V2()
    print("Modelbase created")
    print("Model Ver. C")
    model = modelbase.improved_disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/ResNet50V2_ver_C.h5".format(directory))
    print("Ver. C model saved.")
    del modelbase
    del model
    keras.backend.clear_session()


if __name__ == "__main__":
    test()






        
        
        
        