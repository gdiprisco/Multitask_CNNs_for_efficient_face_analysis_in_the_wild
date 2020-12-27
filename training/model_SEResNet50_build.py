import keras
import os
import tensorflow as tf
from model_abstract import Model, STANDARD_CLASSES, STANDARD_ACT

import sys
# sys.path.append('keras-squeeze-excite-network')
# from keras_squeeze_excite_network import se_resnet
sys.path.append('classification_models')
from classification_models.classification_models.keras import Classifiers
from classification_models.classification_models.models.senet import SEResNetBottleneck as residual_block


class SEResNet50(Model):
    def __init__(self, input_shape=(112, 112, 3), weights=None):
        print("Building SEResNet50", weights if weights is not None else "", input_shape)
        self.input_shape = input_shape
        # self.model = se_resnet.SEResNet50(input_shape=input_shape, weights=weights, include_top=False, pooling='avg', weight_decay=0)
        SEResNet50, _ = Classifiers.get('seresnet50')
        self.model = SEResNet50(input_shape=input_shape, input_tensor=None, weights=weights, include_top=True)
        self._joint_bottom = self.model.layers[-3].output #1
        self._disjoint_bottom = self.model.layers[-60].output #57
        self.buildable = True
        self._available_joint_branches = ["gen1", "age1", "eth1", "emo1", "gen2", "age2", "eth2", "emo2"]
        self._available_disjoint_branches = ["genb", "ageb", "ethb", "emob"]
    
    def _joint_top(self, features, num_classes, activation):
        # features = keras.layers.BatchNormalization(axis=-1)(features)
        # features = keras.layers.Activation('relu')(features)
        current_dense = self._available_joint_branches.pop(0)
        # features = keras.layers.GlobalAveragePooling2D()(features)
        return keras.layers.Dense(num_classes, use_bias=True, activation=activation, name=current_dense)(features)


    def _disjoint_top(self, features, num_classes, activation, improved=False):
        # # SEResNet50 parameters, with bottleneck
        # N = [3, 4, 6, 3] # depth
        # filters = [64, 128, 256, 512]
        # width = 1
        # # appending 4th block
        # k = 3
        # top_features = se_resnet._resnet_bottleneck_block(features, filters[k], width, strides=(2, 2))
        # for _ in range(N[k] - 1):
        #     top_features = se_resnet._resnet_bottleneck_block(top_features, filters[k], width)
        # # appending feature extraction tails
        # top_features = keras.layers.BatchNormalization(axis=-1)(top_features)
        # top_features = keras.layers.Activation('relu')(top_features)
        # # Average Pooling 2D previously removed
        # top_features = keras.layers.GlobalAveragePooling2D()(top_features)
        filters = 2048
        reduction = 16
        groups = 1
        kwargs = Classifiers.get_kwargs()
        top_features = residual_block(filters, reduction=reduction, strides=2, groups=groups, **kwargs)(features)
        top_features = residual_block(filters, reduction=reduction, strides=1, groups=groups, **kwargs)(top_features)
        top_features = residual_block(filters, reduction=reduction, strides=1, groups=groups, **kwargs)(top_features)
        top_features = keras.layers.GlobalAveragePooling2D()(top_features)
        return self._joint_top(top_features, num_classes, activation)

    def _aggregate_low_level_features(self, features):
        # Same solution of the pre-built feature extraction tail
        # aggregate = keras.layers.BatchNormalization(axis=-1, name='post_bn_features')(features)
        # aggregate = keras.layers.Activation('relu', name='post_relu_features')(features)
        # aggregate = keras.layers.GlobalAveragePooling2D()(features)
        return keras.layers.GlobalAveragePooling2D(name='avg_pool_features')(features)
        # print(features.name)
        # exit(1)
        # return features


def test(text_gpu="0"):
    directory = "test_models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(text_gpu)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    # print("Creating modelbase...")
    # modelbase = SEResNet50()
    # print("Modelbase created")
    # print("Model Baseline")
    # model = modelbase.baseline()
    # model.summary()
    # print("Saving model in {}...".format(directory))
    # model.save("{}/SEResNet50_baseline.h5".format(directory))
    # print("Baseline model saved.")
    # # input("Press any key to continue...")
    # del modelbase
    # del model
    # keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = SEResNet50()
    print("Modelbase created")
    print("Model Ver. A")
    model = modelbase.joint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/SEResNet50_ver_A.h5".format(directory))
    print("Ver. A model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = SEResNet50()
    print("Modelbase created")
    print("Model Ver. B")
    model = modelbase.disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/SEResNet50_ver_B.h5".format(directory))
    print("Ver. B model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = SEResNet50()
    print("Modelbase created")
    print("Model Ver. C")
    model = modelbase.improved_disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/SEResNet50_ver_C.h5".format(directory))
    print("Ver. C model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

if __name__ == "__main__":
    test()






        
        
        
        