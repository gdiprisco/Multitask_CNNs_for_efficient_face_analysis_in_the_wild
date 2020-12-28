import numpy as np
from keras.callbacks import Callback

# best_val_acc = 0
# best_val_loss = sys.float_info.max 

# def saveModel(epoch,logs):
#     val_acc = logs['val_acc']
#     val_loss = logs['val_loss']

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         model.save(...)
#     elif val_acc == best_val_acc:
#         if val_loss < best_val_loss:
#             best_val_loss=val_loss
#             model.save(...)

# callbacks = [LambdaCallback(on_epoch_end=saveModel)]


class HistoryMetric(Callback):

    def __init__(self, filepath, monitors):
        super(HistoryMetric, self).__init__()
        self.filepath = filepath
        self.monitors = monitors
        self.best_gender_acc = -np.Inf
        self.best_age_mae = np.Inf
        self.best_ethnicity_acc = -np.Inf
        self.best_emotion_acc = -np.Inf
        self.gen_epoch = self.age_epoch = self.eth_epoch = self.emo_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        gender_acc = logs[self.monitors["gender"]]
        age_mae = logs[self.monitors["age"]]
        ethnicity_acc = logs[self.monitors["ethnicity"]]
        emotion_acc = logs[self.monitors["emotion"]]
        best_found = False

        print("\nEpoch %05d" % (epoch+1))

        if np.greater(gender_acc, self.best_gender_acc):
            print("%s improved from %0.5f to %0.5f" % (self.monitors["gender"], self.best_gender_acc, gender_acc))
            self.best_gender_acc = gender_acc
            self.gen_epoch = epoch + 1
            best_found = True
        else:
            print("%s did not improved from %0.5f of epoch %05d" % (self.monitors["gender"], self.best_gender_acc, self.gen_epoch))

        if np.less(age_mae, self.best_age_mae):
            print("%s improved from %0.5f to %0.5f" % (self.monitors["age"], self.best_age_mae, age_mae))
            self.best_age_mae = age_mae
            self.age_epoch = epoch + 1
            best_found = True
        else:
            print("%s did not improved from %0.5f of epoch %05d" % (self.monitors["age"], self.best_age_mae, self.age_epoch))

        if np.greater(ethnicity_acc, self.best_ethnicity_acc):
            print("%s improved from %0.5f to %0.5f" % (self.monitors["ethnicity"], self.best_ethnicity_acc, ethnicity_acc))
            self.best_ethnicity_acc = ethnicity_acc
            self.eth_epoch = epoch + 1
            best_found = True
        else:
            print("%s did not improved from %0.5f of epoch %05d" % (self.monitors["ethnicity"], self.best_ethnicity_acc, self.eth_epoch))

        if np.greater(emotion_acc, self.best_emotion_acc):
            print("%s improved from %0.5f to %0.5f" % (self.monitors["emotion"], self.best_emotion_acc, emotion_acc))
            self.best_emotion_acc = emotion_acc
            self.emo_epoch = epoch + 1
            best_found = True
        else:
            print("%s did not improved from %0.5f of epoch %05d" % (self.monitors["emotion"], self.best_emotion_acc, self.emo_epoch))
        
        if best_found:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            print("\nSaving model to %s\n" % filepath)
            self.model.save(filepath, overwrite=True)

            

            
        
