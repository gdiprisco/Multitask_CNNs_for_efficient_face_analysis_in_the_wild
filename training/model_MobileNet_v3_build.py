import keras
import os
import tensorflow as tf
from model_abstract import Model, STANDARD_CLASSES, STANDARD_ACT

import sys
sys.path.append("MobileNetV3")
from MobileNetV3 import MobileNetV3


def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

def HSigmoid(x):
    return tf.nn.relu6(x + 3) / 6

imagenet_large_with_top_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MobileNetV3/mobilenet_v3_large_1.0_224_weights.h5")

class MobileNetV3Large(Model):
    def __init__(self, input_shape=(112, 112, 3), weights="imagenet"):
        print("Building MobileNet Large v3", weights if weights is not None else "", input_shape)
        self.input_shape = input_shape
        building_weights = None if weights == 'imagenet' or weights is None else weights
        self.model = MobileNetV3.MobileNetV3_large(input_shape=input_shape, weights=building_weights, include_top=False, pooling='avg')
        if weights == "imagenet":
            self.model.load_weights(imagenet_large_with_top_weights, by_name=True)
        self._joint_bottom = self.model.layers[-1].output
        self._disjoint_bottom = self.model.layers[-49].output # add_8, top of the 12th bottleneck
        self.buildable = True
        self._available_joint_branches = ["gen1", "age1", "eth1", "emo1", "gen2", "age2", "eth2", "emo2"]
        self._available_disjoint_branches = ["genb", "ageb", "ethb", "emob"]
        self._current_branch_name = ""
        self.nlay = 12 #TODO verifica
        self.max_exp_lay_num = 14
    
    def _joint_top(self, features, num_classes, activation):
        current_activation = self._available_joint_branches.pop(0)
        layer = keras.layers.Conv2D(num_classes,(1,1),strides=(1,1),padding='same',use_bias=True)(features)
        layer = keras.layers.Flatten()(layer)
        return keras.layers.Activation(activation, name=current_activation)(layer)

    def _bottleneck_block(self, _inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, is_use_se=True, activation='RE', num_layers=0,ratio=4, *args):
        with tf.name_scope('bottleneck_block'):
            bottleneck_dim = expansion_dim
            channel_axis = -1
            input_shape = keras.backend.int_shape(_inputs)
            r = strides == (1,1) and input_shape[3] == out_dim
            # ** pointwise conv 
            if self.nlay > 0:
                x = self._conv2d_block(_inputs, 
                                        bottleneck_dim,
                                        kernel=(1, 1),
                                        strides=(1, 1),
                                        is_use_bias=is_use_bias,
                                        activation=activation)
            else:
                x = _inputs
            # ** depthwise conv
            x = self._depthwise_block(x,
                                        kernel=kernel, 
                                        strides=strides,
                                        is_use_se=is_use_se,
                                        activation=activation,
                                        num_layers=num_layers,
                                        ratio=ratio)
            # ** pointwise conv
            basename = 'expanded_conv_{}_{}_project'.format(self.nlay, self._current_branch_name)
            x = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name=basename,use_bias=False)(x)
            x = keras.layers.BatchNormalization(axis=channel_axis,name=basename+'_batch_normalization')(x)
            if r:
                x = keras.layers.Add()([x, _inputs])
            self.nlay += 1
        return x

    def _depthwise_block(self, _inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0,ratio=4):
        channel_axis = -1
        basename = 'expanded_conv_{}_{}_depthwise'.format(self.nlay, self._current_branch_name)
        x = keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same',name=basename,use_bias=False)(_inputs)
        x = keras.layers.BatchNormalization(axis=channel_axis,name=basename+'_batch_normalization')(x)
        if activation == 'RE':
            x = keras.layers.ReLU( name=basename+'_activation')(x)
        elif activation == 'HS':
            x = keras.layers.Activation(Hswish, name=basename+'_activation')(x)
        else:
            raise NotImplementedError
        if is_use_se:
            x = self._se_block(x,ratio=ratio)
        return x

    def _conv2d_block(self, _inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', basename=None):
        channel_axis = -1
        if basename is None:
            if self.nlay<0:
                basename = 'conv_{}_{}'.format(self.nlay+1, self._current_branch_name)
            elif self.nlay > self.max_exp_lay_num:
                basename = 'conv_{}_{}'.format(self.nlay-self.max_exp_lay_num, self._current_branch_name)
            else:
                basename = 'expanded_conv_{}_{}_expand'.format(self.nlay, self._current_branch_name)
        x = keras.layers.Conv2D(filters, kernel, strides= strides, padding=padding,use_bias=is_use_bias, name=basename)(_inputs)
        x = keras.layers.BatchNormalization(momentum=0.9,axis=channel_axis,name=basename+'_batch_normalization')(x)
        if activation == 'RE':
            x = keras.layers.ReLU(name=basename+'_activation')(x)
        elif activation == 'HS':
            x = keras.layers.Activation(Hswish, name=basename+'_activation')(x)
        else:
            raise NotImplementedError
        return x

    def _se_block(self, _inputs, ratio=4, pooling_type='avg'):
        channel_axis = -1
        filters = _inputs._keras_shape[channel_axis]
        se_shape = (1, 1, filters)
        
        if pooling_type == 'avg':
            se = keras.layers.GlobalAveragePooling2D()(_inputs)
        elif pooling_type == 'depthwise':
            se = self._global_depthwise_block(_inputs)
        else:
            raise NotImplementedError
        
        se = keras.layers.Reshape(se_shape)(se)
        basename = 'expanded_conv_{}_{}_squeeze_excite_conv_'.format(self.nlay, self._current_branch_name)
        se = keras.layers.Conv2D(int(filters / ratio), (1,1), strides=(1,1), activation='relu', padding='same',use_bias=True, name=basename+'0')(se)
        se = keras.layers.Conv2D(filters, (1,1), strides=(1,1), activation=HSigmoid,padding='same',use_bias=True, name=basename+'1')(se)
        x = keras.layers.multiply([_inputs, se])
        return x

    def _global_depthwise_block(self, _inputs):
        assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
        kernel_size = _inputs._keras_shape[1]
        x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='same')(_inputs)
        return x

    def _disjoint_top(self, features, num_classes, activation, improved=False):
        self.nlay = self.max_exp_lay_num - 2
        self._current_branch_name = self._available_disjoint_branches.pop(0)

        # Debug warning: all parameters here are related to large version of MobileNetV3
        config_12 = MobileNetV3.MobileNetV3.LARGE[-3]
        top_features = self._bottleneck_block(features, *config_12)
        config_13 = MobileNetV3.MobileNetV3.LARGE[-2]
        top_features = self._bottleneck_block(top_features, *config_13)
        config_14 = MobileNetV3.MobileNetV3.LARGE[-1]
        top_features = self._bottleneck_block(top_features, *config_14)

        top_features = self._conv2d_block(top_features,
                                            filters=960, 
                                            kernel=(1, 1), 
                                            strides=(1, 1),
                                            is_use_bias=False,
                                            padding='same',
                                            activation='HS')

        top_features = keras.layers.GlobalAveragePooling2D()(top_features)
        pooled_shape = (1, 1, top_features._keras_shape[-1])
        top_features = keras.layers.Reshape(pooled_shape)(top_features)
        basename = 'conv_2_{}'.format(self._current_branch_name)
        top_features = keras.layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='same', use_bias=True,name=basename)(top_features)
        top_features = keras.layers.Activation(Hswish, name=basename+'_activation')(top_features)
        top_features = self._joint_top(top_features, num_classes, activation)
        return keras.layers.Reshape((1, 1, top_features._keras_shape[-1]))(top_features) if improved else top_features 

    
    def _aggregate_low_level_features(self, features):
        # Same solution of the pre-built feature extraction tail 
        aggregate = self._conv2d_block(features,
                                        filters=960, 
                                        kernel=(1, 1), 
                                        strides=(1, 1),
                                        is_use_bias=False,
                                        padding='same',
                                        activation='HS',
                                        basename='conv_1_aggregate')

        aggregate = keras.layers.GlobalAveragePooling2D()(aggregate)
        pooled_shape = (1, 1, aggregate._keras_shape[-1])
        aggregate = keras.layers.Reshape(pooled_shape)(aggregate)
        aggregate = keras.layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='same', use_bias=True,name='conv_2_aggregate')(aggregate)
        return keras.layers.Activation(Hswish, name='conv_2_aggregate_activation')(aggregate)




def test(text_gpu="0"):
    directory = "test_models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(text_gpu)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

    print("Creating modelbase...")
    modelbase = MobileNetV3Large()
    print("Modelbase created")
    print("Model Baseline")
    model = modelbase.baseline()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/MobileNetV3Large_baseline.h5".format(directory))
    print("Baseline model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = MobileNetV3Large()
    print("Modelbase created")
    print("Model Ver. A")
    model = modelbase.joint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/MobileNetV3Large_ver_A.h5".format(directory))
    print("Ver. A model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = MobileNetV3Large()
    print("Modelbase created")
    print("Model Ver. B")
    model = modelbase.disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/MobileNetV3Large_ver_B.h5".format(directory))
    print("Ver. B model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

    print("Creating modelbase...")
    modelbase = MobileNetV3Large()
    print("Modelbase created")
    print("Model Ver. C")
    model = modelbase.improved_disjoint_extraction_model()
    model.summary()
    print("Saving model in {}...".format(directory))
    model.save("{}/MobileNetV3Large_ver_C.h5".format(directory))
    print("Ver. C model saved.")
    # input("Press any key to continue...")
    del modelbase
    del model
    keras.backend.clear_session()

if __name__ == "__main__":
    test()






        
        
        
        