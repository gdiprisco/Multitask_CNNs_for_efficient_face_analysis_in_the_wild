import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import os
import numpy as np
from glob import glob
import re
import tensorflow as tf
import keras
from keras import backend as K
import time
from center_loss import center_loss
from datetime import datetime
import sys
sys.path.append("../dataset")
from load_dataset import load_dataset
from dataset_utils import MASK_VALUE
from checkpoint_callback import HistoryMetric


def get_modelbase(net, input_shape, weights):
    if net == "resnet50":
        from model_ResNet50_v2_build import ResNet50V2
        modelbase = ResNet50V2(input_shape=input_shape, weights=weights)
    elif net == "seresnet50":
        from model_SEResNet50_build import SEResNet50
        modelbase = SEResNet50(input_shape=input_shape, weights=weights)
    elif net == "mobilenetv3":
        from model_MobileNet_v3_build import MobileNetV3Large
        modelbase = MobileNetV3Large(input_shape=input_shape, weights=weights)
    else:
        raise Exception("Model {} not supported.".format(net))
    return modelbase

def get_modelversioned(modelbase, version, model_name):
    if version == "verA":
        model = modelbase.joint_extraction_model()
    elif version == "verB":
        model = modelbase.disjoint_extraction_model()
    elif version == "verC":
        reshape = model_name == "mobilenetv3"
        model = modelbase.improved_disjoint_extraction_model(reshape=reshape)
    else:
        raise Exception("Version {} not supported.".format(version))
    return model

def get_model(net, input_shape, weights, version):
    return get_modelversioned(get_modelbase(net, input_shape, weights), version, net)


def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return keras.callbacks.LearningRateScheduler(schedule, verbose=1)

def _find_latest_checkpoint(d):
    all_checks = glob(os.path.join(d, '*'))
    max_ep = 0
    max_c = None
    for c in all_checks:
        epoch_num = re.search(ep_re, c)
        if epoch_num is not None:
            epoch_num = int(epoch_num.groups(1)[0])
            if epoch_num > max_ep:
                max_ep = epoch_num
                max_c = c
    return max_ep, max_c

# TODO
def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets
    Args:       loss_function: The loss function to mask
                mask_value: The value to mask in the targets
    Returns:    function: a loss function that acts like loss_function with masked inputs
    """
    def masked_categorical_crossentropy(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    def masked_mean_squared_error(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_se_tensor = (y_true - y_pred) ** 2
        return K.sum(masked_se_tensor) / K.maximum(K.sum(mask), 1)
    
    # def masked_categorical_crossentropy(y_true, y_pred):
    #     return masked_loss_function(y_true, y_pred)

    if loss_function is keras.losses.mean_squared_error:
        return masked_mean_squared_error
    elif loss_function is keras.losses.binary_crossentropy or loss_function is keras.losses.categorical_crossentropy:
        return masked_categorical_crossentropy
    else:
        raise Exception("Masked loss: {} loss not supported.".format(loss_function.__name__))

# def masked_accuracy(y_true, y_pred):
#     total = K.sum(K.not_equal(y_true, MASK_VALUE))
#     correct = K.sum(K.equal(y_true, K.round(y_pred)))
#     return correct / total

def build_masked_acc(acc_function, mask_value=MASK_VALUE):

    def masked_categorical_accuracy(y_true, y_pred): #single_class_accuracy
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        class_y_true = K.argmax(y_true, axis=-1)
        class_y_pred = K.argmax(y_pred, axis=-1)
        mask = K.cast(K.any(mask, axis=-1), K.floatx())
        masked_acc_tensor = K.cast(K.equal(class_y_true, class_y_pred), K.floatx()) * mask
        return K.sum(masked_acc_tensor) / K.maximum(K.sum(mask), 1)


        # class_id_true = K.argmax(y_true, axis=-1)
        # class_id_preds = K.argmax(y_pred, axis=-1)
        # # Replace class_id_preds with class_id_true for recall here
        # accuracy_mask = K.cast(K.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
        # class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        # class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        # return class_acc

    def masked_mean_absolute_error(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_ae_tensor = K.abs(y_true - y_pred)
        return K.sum(masked_ae_tensor) / K.maximum(K.sum(mask), 1)

    if acc_function is keras.metrics.categorical_accuracy:
        return masked_categorical_accuracy
    elif acc_function is keras.metrics.mean_absolute_error:
        return masked_mean_absolute_error
    else:
        raise Exception("Masked accuracy: {} metric not supported.".format(acc_function.__name__))

# def build_masked_accX(acc_function, mask_value=MASK_VALUE):
#     """Builds a loss function that masks based on targets
#     Args:       loss_function: The loss function to mask
#                 mask_value: The value to mask in the targets
#     Returns:    function: a loss function that acts like loss_function with masked inputs
#     """
#     def masked_categorical_accuracy(y_true, y_pred):
#         mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
#         return acc_function(y_true * mask, y_pred * mask)
    
#     def masked_mean_absolute_error(y_true, y_pred):
#         return masked_acc_function(y_true, y_pred)
    
#     def masked_categorical_accuracy(y_true, y_pred):
#         return masked_acc_function(y_true, y_pred)

#     if acc_function == keras.metrics.mean_absolute_error:
#         return masked_mean_absolute_error
#     elif acc_function == keras.metrics.categorical_accuracy:
#         return masked_categorical_accuracy
#     return masked_acc_function

def get_metrics(classes):
    print(classes)
    losses, accuracies = [], []
    for cl in classes:
        if cl == 1:
            losses.append(build_masked_loss(keras.losses.mean_squared_error))
            accuracies.append(build_masked_acc(keras.metrics.mean_absolute_error))
        elif cl > 1:
            losses.append(build_masked_loss(keras.losses.categorical_crossentropy))
            accuracies.append(build_masked_acc(keras.metrics.categorical_accuracy))
    if len(losses) == 1:
        losses = losses[0]
    return losses, accuracies
    

def evalds(part, dataset, model):
            dataset_test = dataset(part, target_shape=input_shape, augment=False, preprocessing=args.preprocessing)
            print('Evaluating %s results...' % part)
            result = model.evaluate_generator(dataset_test.get_generator(batch_size), verbose=1, workers=4)
            print('%s results: loss %.3f - accuracy %.3f' % (part, result[0], result[1]))

# available parameters
available_dataset = "VGGFace2-RAF"
available_nets = ['resnet50', 'seresnet50', 'mobilenetv3']
available_versions = ["verA", "verB", "verC"]
available_normalizations = ['z_normalization', 'full_normalization', 'vggface2']
available_augmentations = ['default', 'vggface2', 'autoaugment-rafdb', 'no']
available_modes = ['train', 'training', 'test', 'train_inference', 'test_inference']

# loss = {
#     "gen1" : build_masked_loss(keras.losses.binary_crossentropy),
#     # "age1" : build_masked_loss(keras.losses.binary_crossentropy),
#     "age1" : build_masked_loss(keras.losses.mean_squared_error),
#     "eth1" : build_masked_loss(keras.losses.binary_crossentropy),
#     "emo1" : build_masked_loss(keras.losses.binary_crossentropy)
# }

# # loss = {
# #     "gen1" : build_masked_loss(keras.losses.categorical_crossentropy),
# #     # "age1" : build_masked_loss(keras.losses.binary_crossentropy),
# #     "age1" : build_masked_loss(keras.losses.mean_squared_error),
# #     "eth1" : build_masked_loss(keras.losses.categorical_crossentropy),
# #     "emo1" : build_masked_loss(keras.losses.categorical_crossentropy)
# # }

# loss_weights = {
#     "gen1" : 10.0,
#     # "age1" : 1.0,
#     "age1" : 0.025,
#     "eth1" : 10.0,
#     "emo1" : 20.0 # 50.0 # 100.0
# }

# accuracy = {
#     "gen1" : build_masked_acc(keras.metrics.categorical_accuracy),
#     # "age1" : build_masked_acc(keras.metrics.categorical_accuracy),
#     "age1" : build_masked_acc(keras.metrics.mean_absolute_error),
#     "eth1" : build_masked_acc(keras.metrics.categorical_accuracy),
#     "emo1" : build_masked_acc(keras.metrics.categorical_accuracy)
# }

loss1 = {
    "gen1" : build_masked_loss(keras.losses.binary_crossentropy),
    "age1" : build_masked_loss(keras.losses.mean_squared_error),
    "eth1" : build_masked_loss(keras.losses.binary_crossentropy),
    "emo1" : build_masked_loss(keras.losses.binary_crossentropy),
}

loss2 = {
    "gen2" : build_masked_loss(keras.losses.binary_crossentropy),
    "age2" : build_masked_loss(keras.losses.mean_squared_error),
    "eth2" : build_masked_loss(keras.losses.binary_crossentropy),
    "emo2" : build_masked_loss(keras.losses.binary_crossentropy)
}

loss_weights1 = {
    "gen1" : 10.0,
    "age1" : 0.025,
    "eth1" : 10.0,
    "emo1" : 50.0,
}

loss_weights2 = {
    "gen2" : 10.0,
    "age2" : 0.025,
    "eth2" : 10.0,
    "emo2" : 50.0
}

accuracy1 = {
    "gen1" : build_masked_acc(keras.metrics.categorical_accuracy),
    "age1" : build_masked_acc(keras.metrics.mean_absolute_error),
    "eth1" : build_masked_acc(keras.metrics.categorical_accuracy),
    "emo1" : build_masked_acc(keras.metrics.categorical_accuracy),
}

accuracy2 = {
    "gen2" : build_masked_acc(keras.metrics.categorical_accuracy),
    "age2" : build_masked_acc(keras.metrics.mean_absolute_error),
    "eth2" : build_masked_acc(keras.metrics.categorical_accuracy),
    "emo2" : build_masked_acc(keras.metrics.categorical_accuracy)
}

MONITORED_METRIC = "val_emo{}_masked_categorical_accuracy"

def get_versioned_metrics(version):
    if version in available_versions[0:2]:
        return loss1, loss_weights1, accuracy1, 1
    elif version == available_versions[2]:
        return {**loss1, **loss2}, {**loss_weights1, **loss_weights2}, {**accuracy1, **accuracy2}, 2
        # return loss2, loss_weights2, accuracy2, 2
    else:
        raise Exception("Version {} not supported: unable to get right losses and accuracies".format(version)) 

if __name__ == "__main__":
    
    # available_lpf = [0, 1, 2, 3, 5, 7]

    # parser of python arguments
    parser = argparse.ArgumentParser(description='Common training and evaluation.')
    # parser.add_argument('--lpf', dest='lpf_size', type=int, choices=available_lpf, default=1, help='size of the lpf filter (1 means no filtering)')
    parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
    # parser.add_argument('--center_loss', action='store_true', help='use center loss')
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
    parser.add_argument('--momentum', action='store_true')
    parser.add_argument('--dataset', dest='dataset', type=str, default=available_dataset, help='dataset to use for the training')
    parser.add_argument('--mode', dest='mode', type=str,choices=available_modes, default='train', help='train or test')
    parser.add_argument('--epoch', dest='test_epoch', type=int, default=None, help='epoch to be used for testing, mandatory if mode=test')
    parser.add_argument('--training-epochs', dest='n_training_epochs', type=int, default=220, help='epoch to be used for training, default 220')
    parser.add_argument('--dir', dest='dir', type=str, default=None, help='directory for resuming train and reading/writing training data and logs')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
    parser.add_argument('--sel_gpu', dest='selected_gpu', type=str, default="0", help="one number or two numbers separated by a hyphen")
    parser.add_argument('--input_shape', dest='input_shape', type=str, default="112,112,3", help="input shape in the format 'w,h,c'")
    parser.add_argument('--net', type=str, choices=available_nets, help='Network architecture')
    parser.add_argument('--version', type=str, choices=available_versions, help='Network architecture version')
    parser.add_argument('--resume', action='store_true', help='resume training from DIR argument')
    parser.add_argument('--pretraining', type=str, default=None, help='Pretraining weights, do not set for None, can be imagenet or a file')
    parser.add_argument('--preprocessing', type=str, default='full_normalization', choices=available_normalizations)
    parser.add_argument('--augmentation', type=str, default='default', choices=available_augmentations)
    args = parser.parse_args()

    # importing dataset
    Dataset = load_dataset(args.dataset)

    # learning Rate
    lr = args.lr.split(':')
    initial_learning_rate = float(lr[0])  # 0.002
    learning_rate_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
    learning_rate_decay_epochs = int(lr[2]) if len(lr) > 2 else 40

    # Epochs to train
    n_training_epochs = args.n_training_epochs

    # Batch size
    batch_size = args.batch_size

    # Model building
    input_shape = tuple(int(s) for s in args.input_shape.split(","))
    if len(input_shape) != 3:
        raise Exception("Input shape must contain 3 elements. E.g. '112,112,3'")
    elif input_shape[0] != input_shape[1]:
        raise Exception("Input shape must be in format 'channel last'. E.g. '112,112,3'")

    # Model loading
    gpu_to_use = [str(s) for s in args.selected_gpu.split(',') if s.isdigit()]
    if args.ngpus <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
        # model, feature_layer = get_model(args.net, input_shape, args.pretraining, args.version)
        model = get_model(args.net, input_shape, args.pretraining, args.version)
    else:
        if len(gpu_to_use) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
        print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
        print("WARNING: Using %d gpus" % args.ngpus)
        with tf.device('/cpu:0'):
            # model, feature_layer = get_model(args.net, input_shape, args.pretraining, args.version)
            model = get_model(args.net, input_shape, args.pretraining, args.version)
        model = keras.utils.multi_gpu_model(model, args.ngpus)
    model.summary()

    # model compiling
    if args.weight_decay:
        weight_decay = args.weight_decay  # 0.0005
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, keras.layers.DepthwiseConv2D) or isinstance(
                    layer, keras.layers.Dense):
                layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))
    # optimizer = keras.optimizers.sgd(momentum=0.9) if args.momentum else 'sgd'
    optimizer = "adam"

    #TODO
    # optimizer = 'rmsprop'

    # TODO
    # if args.center_loss:
    #     loss = center_loss(feature_layer, keras.losses.categorical_crossentropy, 0.9, NUM_CLASSES, 0.01, features_dim=2048)
    # else:
    #     loss = keras.losses.categorical_crossentropy if NUM_CLASSES > 1 else keras.losses.mean_squared_error
    # accuracy_metrics = [keras.metrics.categorical_accuracy] if NUM_CLASSES > 1 else [keras.metrics.mean_squared_error]
    # model.compile(loss=loss, optimizer=optimizer, metrics=accuracy_metrics)

    # loss, accuracy = get_metrics(Dataset.get_num_classes())
    # print("Losses:", [l.__name__ for l in loss])
    # print("Accuracies:", [a.__name__ for a in accuracy])
    # # exit()

    
    loss, loss_weights, accuracy, metric_version = get_versioned_metrics(args.version)
    model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=accuracy)

    monitors = {
        "gender" : "val_gen{}_masked_categorical_accuracy".format(metric_version),
        "age" : "age{}_masked_mean_absolute_error".format(metric_version),
        "ethnicity" : "val_eth{}_masked_categorical_accuracy".format(metric_version),
        "emotion" : "val_emo{}_masked_categorical_accuracy".format(metric_version),
    }

    # Directory creating to store model checkpoints
    datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
    dirnm = "inference_time_test" if args.mode.endswith('inference') else "trained"
    dirnm = os.path.join("..", dirnm)
    if not os.path.isdir(dirnm): os.mkdir(dirnm)
    argstring = ''.join(sys.argv[1:]).replace('--', '_').replace('=', '').replace(':', '_')
    dirnm += '/%s_%s' % (argstring, datetime)
    if args.cutout: dirnm += '_cutout'
    if args.dir: dirnm = args.dir
    if not os.path.isdir(dirnm): os.mkdir(dirnm)
    filepath = os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
    logdir = dirnm
    ep_re = re.compile('checkpoint.([0-9]+).hdf5')

    # AUGMENTATION 
    if args.cutout:
        from cropout_test import CropoutAugmentation
        custom_augmentation = CropoutAugmentation()
    elif args.augmentation == 'autoaugment-rafdb':
        from autoaug_test import MyAutoAugmentation
        from autoaugment.rafdb_policies import rafdb_policies
        custom_augmentation = MyAutoAugmentation(rafdb_policies)
    elif args.augmentation == 'default':
        from dataset_tools import DefaultAugmentation
        custom_augmentation = DefaultAugmentation()
    elif args.augmentation == 'vggface2':
        from dataset_tools import VGGFace2Augmentation
        custom_augmentation = VGGFace2Augmentation()
    else:
        custom_augmentation = None


    if args.mode.startswith('train'):
        print("TRAINING %s" % dirnm)
        dataset_training = Dataset('train', target_shape=input_shape, augment=True, preprocessing=args.preprocessing, custom_augmentation=custom_augmentation)
        dataset_validation = Dataset('val', target_shape=input_shape, augment=False, preprocessing=args.preprocessing)

        lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)
        
        # checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False)
        # MONITORED_METRIC = MONITORED_METRIC.format(2 if args.version == "verC" else 1)
        # checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, monitor=MONITORED_METRIC)
        
        checkpoint = HistoryMetric(filepath=filepath, monitors=monitors)
        tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
        callbacks_list = [lr_sched, checkpoint, tbCallBack]

        if args.resume:
            pattern = filepath.replace('{epoch:02d}', '*')
            epochs = glob(pattern)
            print(pattern)
            print(epochs)
            epochs = [int(x[-8:-5].replace('.', '')) for x in epochs]
            initial_epoch = max(epochs)
            print('Resuming from epoch %d...' % initial_epoch)
            model.load_weights(filepath.format(epoch=initial_epoch))
        else:
            initial_epoch = 0

        if args.mode == "train_inference":
            batch_size = 1
            print("Warning: TEST ON CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = ''

        doublelabel = metric_version == 2
        train_generator = dataset_training.get_generator(batch_size, doublelabel=doublelabel)
        validation_generator = dataset_validation.get_generator(batch_size, doublelabel=doublelabel)
        model.fit_generator(generator=train_generator,
                            validation_data=validation_generator,
                            verbose=1,
                            callbacks=callbacks_list,
                            epochs=n_training_epochs,
                            workers=8,
                            initial_epoch=initial_epoch)
    elif args.mode == 'test':
        if args.test_epoch is None:
            args.test_epoch, _ = _find_latest_checkpoint(dirnm)
            print("Using epoch %d" % args.test_epoch)
        model.load_weights(filepath.format(epoch=int(args.test_epoch)))
        evalds('test', Dataset, model)
        evalds('val', Dataset, model)
        evalds('train', Dataset, model)

