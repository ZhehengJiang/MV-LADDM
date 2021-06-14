
from keras.utils.np_utils import *
import numpy as np
import pandas as pd

# Hyperparams
max_length=50 # Maximum length of the sentence
def feature_connect(a_time,a_force):
    a=np.array([])
    for j in range(int(11340/15)):
        f=np.array([])
        for i in range(15):
            f = np.concatenate((f, a_force[j*15+i,:]), axis=0) if f.size else a_force[j*15+i,:]
        a=np.c_[a,f] if a.size else f    
    return np.c_[a_time,np.transpose(a)],np.transpose(a)

class Dataloader:
    
    def __init__(self):

        self.max_l = max_length
        self.num_classes = 13
        print("Labels used for this classification: ", [1,2,3,4,5,6,7,8,9,10,11,12,13])

    def get_one_hot(self, label):
        label_arr = [0]*self.num_classes
        label_arr[label]=1
        return label_arr[:]
    
    def load_MVLADDM_data(self,):
        
        self.x1 = pd.read_csv('../data/feature_side_train.csv',dtype=np.float32).values
        self.x1 = self.x1.reshape(self.x1.shape[0],self.x1.shape[1],1)
        self.x2 = pd.read_csv('../data/feature_top_train.csv',dtype=np.float32).values
        self.x2 = self.x2.reshape(self.x2.shape[0],self.x2.shape[1],1)
        self.label = pd.read_csv('../data/labels_train.csv',dtype=np.int32).values
        self.label = to_categorical(self.label-1)

#    test    
        self.x1_test =pd.read_csv('../data/feature_side_test.csv',dtype=np.float32).values
        self.x1_test = self.x1_test.reshape(self.x1_test.shape[0], self.x1_test.shape[1], 1)
        self.x2_test = pd.read_csv('../data/feature_top_test.csv',dtype=np.float32).values
        self.x2_test = self.x2_test.reshape(self.x2_test.shape[0], self.x2_test.shape[1], 1)
        self.label_test = pd.read_csv('../data/labels_test.csv',dtype=np.int32).values
        self.label_test = to_categorical(self.label_test-1)

