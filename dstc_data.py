
import numpy as np
import random
import pickle
import pdb
#from autoencoder_test import AutoEncoderApplier
# ==========
#   DATA
# ==========
class DstcData(object):
    def __init__(self, slot_name, one_hot=False, multi_label=False, is_encode=False):
        #=================load feature file========================
        with open('./b_features', 'rb') as fr:
            input_data = pickle.load(fr)
        
        #=================Get max_n_steps, max_turn_len========================
        self.max_n_steps = 0 # the length of input is unfixed
        self.max_turn_len = 0 # the length of input is unfixed
        for (session_id, data) in input_data.items():
            tmp_list = []
            steps = 0
            for each_turn_feature in data[slot_name]:
                steps += 1
                if (len(each_turn_feature) > self.max_turn_len):
                    self.max_turn_len = len(each_turn_feature)
            if (steps > self.max_n_steps):
                self.max_n_steps = steps

        #=================Get slot features========================
        self.slot_features = []
        self.seqlen = []
        for (session_id, data) in input_data.items():
            tmp_list = []
            for each_turn_feature in data[slot_name]:
                tmp_list.append(each_turn_feature)
            self.seqlen.append(len(tmp_list))
            if (len(tmp_list) < self.max_n_steps):
                for i in range(self.max_n_steps - len(tmp_list)):
                    tmp_list.append([0.0] * self.max_turn_len) # padding 
            self.slot_features.append(tmp_list)
        
        #pdb.set_trace()
        
        #=================Load label========================
        self.slot_label = []
        with open('./b_labels', 'rb') as fr:
            labels = pickle.load(fr)
        if (multi_label):
            for (session_id, dialog_label) in labels.items():
                #print ("process session " + session_id)
                tmp_list_0 = []
                for each_turn_label in dialog_label:
                    tmp_list_1 = []
                    for (value, p) in each_turn_label[slot_name].items():
                        tmp_list_1.append(p)
                    tmp_list_0.append(tmp_list_1)
                self.slot_label.append(tmp_list_0)
        else :
            for (session_id, dialog_label) in labels.items():
                tmp_list = []
                for (value, p) in dialog_label[-1][slot_name].items():
                    tmp_list.append(p)
                self.n_classes = len(tmp_list)
                self.slot_label.append(tmp_list)


        if (one_hot):
            self.slot_features = np.array(self.slot_features).reshape(len(self.slot_features), (self.max_turn_len * self.max_n_steps)).tolist()
        
        #=================encode========================
        #if (is_encode):
        #    one_hot_feature = np.array(self.slot_features).reshape(len(self.slot_features), (self.max_turn_len * self.max_n_steps)).tolist()
        #    encoded_feature = AutoEncoderApplier(one_hot_feature, './autoencoder_100.tfmodel').encode_decode
        #    self.slot_features = encoded_feature.reshape(len(self.slot_features), self.max_n_steps, self.max_turn_len).tolist()

    def next_batch(self, batch_size):
        input_arr = []
        label_arr = []
        seq_len_arr = []
        for i in range(batch_size):
            random_num = random.randint(0, len(self.slot_features) - 1) # 0 <= int <= len-1
            input_arr.append(self.slot_features[random_num])
            label_arr.append(self.slot_label[random_num])
            seq_len_arr.append(self.seqlen[random_num])
        return input_arr, label_arr, seq_len_arr


