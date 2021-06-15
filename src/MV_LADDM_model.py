
# keras = tf.contrib.keras
from keras.layers import Input, Dense, Lambda, LSTM, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from src.label_correlation_layer import label_correlation_layer,lcl_loss
from src.data_helpers import Dataloader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from src.MLVAE_layer import MLVAE
import numpy as np

def concatenate_fusion(ID, text, audio):
    modals = []
    for vid, utts in ID.items():
        modals.append(np.concatenate((text[vid], audio[vid]), axis=1))
    return np.array(modals)

class MV_LADDM:

    def __init__(self,classes, MLVAE_latent_dim, num_cameras, sequence_length):
        self.epochs = 1000
        self.batch_size = 1
        self.LSTM_dim = 50
        self.classes = classes
        self.MLVAE_latent_dim = MLVAE_latent_dim
        self.num_channels = num_cameras
        self.sequence_length = sequence_length
        self.lcl = label_correlation_layer(self.classes)
        self.MLVAE = MLVAE(feature_dim=2*self.LSTM_dim, latent_dim=self.MLVAE_latent_dim, num_channels=self.num_channels, sequence_length=self.sequence_length)
        self.PATH = "./models/weights.hdf5"
        self.OUTPUT_PATH = "./pickles/output.pkl"
        print("Model initiated for classification")

    def load_data(self):

        print('Loading data')
        self.data = Dataloader()

        self.data.load_MVLADDM_data()
        self.train_x = None
        self.train_y = None
        for i in range(0, self.data.x_train.shape[0] - self.sequence_length, 80):
            if not self.train_x is None:
                self.train_x = np.concatenate((self.train_x, self.data.x_train[i:i + self.sequence_length, :].reshape(
                                                   1, self.sequence_length, self.data.x_train.shape[1],
                                                   self.data.x_train.shape[2])), axis=0)
                self.train_y = np.concatenate((self.train_y,
                                               self.data.label_train[i:i + self.sequence_length, :].reshape(1,
                                                     self.sequence_length,
                                                     self.data.label_train.shape[
                                                         1])),axis=0)
            else:
                self.train_x = self.data.x_train[i:i + self.sequence_length, :].reshape(1,
                                            self.sequence_length, self.data.x_train.shape[1],
                                            self.data.x_train.shape[2])
                self.train_y = self.data.label_train[i:i + self.sequence_length, :].reshape(1,
                                             self.sequence_length, self.data.label_train.shape[1])
        self.val_x = self.data.x_test
        num_seq_val = int(self.val_x.shape[0] / self.sequence_length)
        self.val_x = self.val_x[0:num_seq_val * self.sequence_length, :, :]
        self.val_x = self.val_x.reshape(num_seq_val, self.sequence_length, self.val_x.shape[1], self.val_x.shape[2])
        self.test_x = self.data.x_test
        num_seq_test = int(self.test_x.shape[0] / self.sequence_length)
        self.test_x = self.test_x[0:num_seq_test * self.sequence_length, :, :]
        self.test_x = self.test_x.reshape(num_seq_test, self.sequence_length, self.test_x.shape[1], self.test_x.shape[2])
        self.val_y = self.data.label_train
        self.val_y = self.val_y[0:num_seq_val * self.sequence_length, :]
        self.val_y = self.val_y.reshape(num_seq_val, self.sequence_length, self.val_y.shape[1])
        self.test_y = self.data.label_test
        self.test_y = self.test_y[0:num_seq_test * self.sequence_length, :]
        self.test_y = self.test_y.reshape(num_seq_test, self.sequence_length, self.test_y.shape[1])

    def calc_test_result(self, pred_label, test_label):

        true_label = []
        predicted_label = []
        pred_label = pred_label.reshape(pred_label.shape[0] * pred_label.shape[1], pred_label.shape[2])
        test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], test_label.shape[2])
        np.savetxt("pred_label.csv", pred_label, delimiter=",")
        np.savetxt("test_label.csv", test_label, delimiter=",")
        for i in range(pred_label.shape[0]):
            for j in range(pred_label.shape[1]):
                true_label.append(np.argmax(test_label[i, :]))
                predicted_label.append(np.argmax(pred_label[i, :]))
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=4))
        print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))

    def get_MVLADDM_model(self):

        # Modality specific parameters
        self.embedding_dim = self.train_x.shape[2]
        print("Creating Model...")

        inputs = Input(shape=(self.sequence_length, self.embedding_dim, self.num_channels), dtype='float32')
        labels = Input(shape=(self.sequence_length, self.classes), dtype='float32')
        branch_outputs = []
        for i in range(self.num_channels):
            out = Lambda(lambda x: x[:, :, :, i])(inputs)
            out = Bidirectional(LSTM(50, activation='tanh', return_sequences=True, dropout=0.4))(out)
            branch_outputs.append(out)

        # MLVAE_layer
        merge, elbo_loss = self.MLVAE(branch_outputs)
        ### label_correlation_layer
        output = TimeDistributed(Dense(50, activation='relu'))(merge)
        output = self.lcl(output)
        model = Model([inputs, labels],output)
        self.loss_function = lcl_loss(labels,output)
        self.elbo_loss = elbo_loss
        self.total_loss = self.loss_function+self.elbo_loss
        return model

    def train_model(self):

        checkpoint = ModelCheckpoint(self.PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        model = self.get_MVLADDM_model()
        model.add_loss(self.total_loss)
        model.compile(optimizer='adam')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit([self.train_x, self.train_y],
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  #		                sample_weight=self.train_mask,
                  shuffle=True,
                  callbacks=[early_stopping, checkpoint],
                  validation_data=([self.val_x, self.val_y],None))

        self.test_model()

    def test_model(self):

        model = load_model(self.PATH)

        self.calc_test_result(model.predict(self.test_x), self.test_y)

