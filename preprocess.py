# dialog_acts example:
# "dialog-acts": [
#                     {
#                         "slots": [
#                             [
#                                 "food", 
#                                 "french"
#                             ]
#                         ], 
#                         "act": "select"
#                     }, 
#                     {
#                         "slots": [
#                             [
#                                 "food", 
#                                 "german"
#                             ]
#                         ], 
#                         "act": "select"
#                     }
#                 ]

from time import clock
import json
import pdb 
import pickle
import sys
from pprint import pprint

prefix_path = './'
data_path = prefix_path + 'data/'
config_path = prefix_path + 'config/'
train_file_list = 'dstc2_train.flist'
ontology_file = 'ontology_dstc2.json'

def print_train_ele(filename):
    # read json file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # parse json
    turns = data['turns']
    for turn in turns:
        print ("dialog acts is %s" % turn['output']['dialog-acts'])
        print ("usr input is <%s>" % turn['input']['live']['asr-hyps'][0]['asr-hyp'])

def get_ontology_info():
    slots_list = []
    slot_values_dict = {}
    with open(config_path + ontology_file, 'r') as f:
        data = json.load(f)
    for (k, v) in data['informable'].items():
        slots_list.append(k)  
    for slot in slots_list:
        slot_values_dict[slot] = data['informable'][slot]
    return slots_list, slot_values_dict

def get_nbest_asr(asr_hyps, n):
    for item in asr_hyps:
        # example of item : {'asr-hyp': 'all', 'score': -4.49337}
        if (len(item['asr-hyp']) > 0):
            return item['asr-hyp']
    return ''

def get_session_id(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    f.close()
    return data['session-id']

def load_input(filename):
    usr_input = []
    dialog_acts = []
    # read json
    with open(filename, 'r') as f:
        data = json.load(f)
    # parse json
    turns = data['turns']
    for turn in turns:
        #usr_input.append(turn['input']['live']['asr-hyps'][0]['asr-hyp']) # load user input, get top 1 asr hyp
        usr_input.append(get_nbest_asr(turn['input']['live']['asr-hyps'], 1))
        dialog_acts.append(turn['output']['dialog-acts'])
    f.close()
    return usr_input, dialog_acts

def load_label(filename):
    label_data = []
    # read json
    with open(filename, 'r') as f:
        data = json.load(f)
    # parse json
    turns = data['turns']
    for turn in turns:
        label_data.append(turn['goal-labels'])
    return label_data

def get_unigram(usr_input):
    ret_list = []
    ret_list = usr_input.split(' ')
    return ret_list

def get_bigram(unigrams):
    ret_list = []
    for i in range(len(unigrams) - 1):
        ret_list.append(unigrams[i] + '_' + unigrams[i+1])
    return ret_list

def get_asr_ngrams(usr_input, n, flag = False):
    # ret_dict: key is ngram text, value is score
    ret_dict = {}
    unigrams = get_unigram(usr_input)
    bigrams = get_bigram(unigrams)
    if (n == 1):
        all_grams = unigrams
    elif (n == 2):
        all_grams = unigrams + bigrams
    for item in all_grams:
        ret_dict[item] = 1.0 
    if (flag):
        return ret_dict.keys()
    return ret_dict

def get_m_act_grams(dialog_act, flag = False):
    # ret_dict: key is ngram text, value is score
    # Assume that there is only one slot for each action in train set
    # it is possible that there are more than one action in one turn
    ret_dict = {}
    for item in dialog_act:
        act = item['act']
        ret_dict[act] = 1.0
        if (len(item['slots']) > 0):
            slot = item['slots'][0][0]
            value = item['slots'][0][1]
            ret_dict[slot] = 1.0
            ret_dict[value] = 1.0
            ret_dict[act+'_'+slot] = 1.0
            ret_dict[slot+'_'+value] = 1.0
            ret_dict[act+'_'+slot+'_'+value] = 1.0
    if (flag):
        return ret_dict.keys()
    return ret_dict

''' tag both <slot> and <value>'''
def tag_ngrams_list(ngrams, slot, values):
    ret_dict = {}
    for ngram in ngrams:
        words = ngram.split('_')

        flag = 0 
        tmp_words = words
        for i in range(len(tmp_words)):
            if (tmp_words[i] == slot):
                flag = 1
                tmp_words[i] = '<slot>'
        tagged_ngram = '_'.join(tmp_words)
        if (flag == 1):
            ret_dict[tagged_ngram] = 1.0

        flag = 0
        tmp_words = words
        for i in range(len(tmp_words)):
            if (tmp_words[i] in values):
                flag = 1
                tmp_words[i] = '<value>'
        tagged_ngram = '_'.join(tmp_words)
        if (flag == 1):
            ret_dict[tagged_ngram] = 1.0

        flag = 0
        tmp_words = words
        for i in range(len(tmp_words)):
            if (tmp_words[i] == slot):
                flag = 1
                tmp_words[i] = '<slot>'
            if (tmp_words[i] in values):
                flag = 1
                tmp_words[i] = '<value>'
        tagged_ngram = '_'.join(tmp_words)
        if (flag == 1):
            ret_dict[tagged_ngram] = 1.0


    return ret_dict.keys()

''' only tag <value>'''
def tag_ngrams_list_by_value(ngrams, value):
    ret_dict = {}
    for ngram in ngrams:
        words = ngram.split('_')
        flag = 0 
        for i in range(len(words)):
            if (words[i] == value):
                flag = 1
                words[i] = '<value>'
        tagged_ngram = '_'.join(words)
        if (flag == 1):
            ret_dict[tagged_ngram] = 1.0

    return ret_dict.keys()

def get_feature(usr_inputs, dialog_acts, f_ngram_list, fs_ngram_dict, fv_ngram_dict, slot_values_dict):
    #ret_dict = init_feature(f_ngram_list, fs_ngram_dict, fv_ngram_dict, slot_values_dict) 
    ret_dict = {}

    # get ngram
    asr_ngrams = [] # ngrams for each turn
    m_act_grams = [] # ngrams for each turn

    for usr_input in usr_inputs:
        asr_ngrams.append(list(get_asr_ngrams(usr_input, 2, True))) # get 1-gram, 2-gram item in usr input
    for dialog_act in dialog_acts:
        m_act_grams.append(list(get_m_act_grams(dialog_act, True)))

    # init ret_dict
    for (slot, values) in slot_values_dict.items():
        ret_dict[slot] = []

    # calculate f, fs, fv
    # asr_ngram is ngram for one turn
    initial_feature = init_feature(f_ngram_list, fs_ngram_dict, fv_ngram_dict, slot_values_dict)
    for (each_turn_asr_ngram, each_turn_m_act_gram) in zip(asr_ngrams, m_act_grams):
        tmp_list = initial_feature
        f_ngram = each_turn_asr_ngram + each_turn_m_act_gram
        f_index = get_ngram_idx(f_ngram, f_ngram_dict)

        
        fs_ngram = get_fs_ngram(f_ngram, slot_values_dict)
        fv_ngram = get_fv_ngram(f_ngram, slot_values_dict)
        for slot in slot_values_dict.keys():
            fs_index = get_ngram_idx(fs_ngram[slot], fs_ngram_dict[slot])
            fv_index = get_ngram_idx(fv_ngram[slot], fv_ngram_dict[slot])
            
            for i in f_index:
                tmp_list[slot][i] = 1.0
            for i in fs_index:
                tmp_list[slot][i] = 1.0
            for i in fv_index:
                tmp_list[slot][i] = 1.0

        for slot in tmp_list.keys():
            ret_dict[slot].append(tmp_list[slot])
    return ret_dict

def is_in_label(label, slot, value):
    for (k, v) in label.items():
        if (k == slot and v == value):
            return True
        else:
            return False

def format_label(lables, slot_values_dict):
    ret_list = []

    tmp_dict = {}
    for each_turn_label in lables: # lable is a dict, key is slot name, value is value of slot
        #pprint(each_turn_label)
        #for (slot, values) in slot_values_dict.items():
        #    tmp_dict[slot] = {}

        for (slot, values) in slot_values_dict.items():
            tmp_dict[slot] = {}
            for value in values:
                if (is_in_label(each_turn_label, slot, value)):
                    tmp_dict[slot][value]  = 1.0
                else:
                    tmp_dict[slot][value]  = 0.0
        ret_list.append(tmp_dict)
    return ret_list
    
def get_ngram(usr_inputs, dialog_acts):
    asr_ngrams = []
    m_act_grams = []
    for usr_input in usr_inputs:
        asr_ngrams += get_asr_ngrams(usr_input, 2, True) # get 1-gram, 2-gram item in usr input
    for dialog_act in dialog_acts:
        m_act_grams += get_m_act_grams(dialog_act, True)
    return asr_ngrams, m_act_grams

def get_fs_ngram(ngram, slot_values_dict):
    ret_dict = {} # key is slot, value is ngram list
    for (slot, values) in slot_values_dict.items():
        tagged_ngram = tag_ngrams_list(ngram, slot, values)
        ret_dict[slot] = list(tagged_ngram)
    return ret_dict

def get_fv_ngram(ngram, slot_values_dict):
    ret_dict = {} # key is slot, value is ngram list
    for (slot, values) in slot_values_dict.items():
        tmp_list = []
        for value in values:
            tagged_ngram = tag_ngrams_list_by_value(ngram, value)
            tmp_list += list(tagged_ngram)
        ret_dict[slot] = tmp_list
    return ret_dict

def merge_dict(dst_dict, src_dict):
    '''
    both of two dicts : key is str, value is list
    func: merge list for the same key
    '''
    for (k, v) in src_dict.items(): 
        tmp_list = list(set(dst_dict[k] + v)) 
        dst_dict[k] = tmp_list
    return dst_dict

def init_dict(slot_values_dict):
    ret_dict = {}
    for s in slot_values_dict.keys():
        ret_dict[s] = []
    return ret_dict

def init_feature(f_ngram_list, fs_ngram_dict, fv_ngram_dict, slot_values_dict):
    ret_dict = {}
    for slot in slot_values_dict.keys():
        tmp_list = [0.0] * (len(f_ngram_list) + len(fs_ngram_dict[slot]) + len(fv_ngram_dict[slot]))
        ret_dict[slot] = tmp_list
    return ret_dict

def get_ngram_idx(ngram_list, dict_list):
    ret_list = []
    for item in ngram_list:
        ret_list.append(dict_list.index(item))
    return ret_list

if __name__ =="__main__":

    # read train set config
    fr = open(config_path + train_file_list, 'r')
    filelist = fr.readlines()
    fr.close()
    #print (filelist)
    
    # Read ontology info
    slots_list, slot_values_dict = get_ontology_info() 

#=============load ngram dict begin============
    dialog_data = {}
    f_ngram_dict = []
    fs_ngram_dict = init_dict(slot_values_dict)
    fv_ngram_dict = init_dict(slot_values_dict)
    # Build f ngram dict
    for filename in filelist:
        filename = filename.strip()
        log_file = data_path + filename + '/log.json' # train file

        print (log_file)
        usr_inputs, dialog_acts = load_input(log_file)

        # save dialog data
        session_id = get_session_id(log_file) 
        dialog_data[session_id] = []
        dialog_data[session_id].append(usr_inputs)
        dialog_data[session_id].append(dialog_acts)

        # get f ngram
        asr_ngram_list, dlg_acts_ngram_list = get_ngram(usr_inputs, dialog_acts)
        f_ngram_dict += asr_ngram_list + dlg_acts_ngram_list

    f_ngram_dict = list(set(f_ngram_dict))
    
    # Build fv, fs ngram
    fs_ngram_dict = merge_dict(fs_ngram_dict, get_fs_ngram(f_ngram_dict, slot_values_dict))
    fv_ngram_dict = merge_dict(fv_ngram_dict, get_fv_ngram(f_ngram_dict, slot_values_dict))

   
#=============get feature============
    input_features = {}
    for (session_id, _) in dialog_data.items():
        print ("get feature for session_id [%s]..." % session_id)
        '''
        Load data
        [usr_inputs] : a list of user input of each turn in a dialog
        [dialog_acts] : a list of dialog actions of each turn in a dialog
        the example of dialog_acts is in the head of this script
        '''
        usr_inputs = _[0]
        dialog_acts = _[1]
        input_features[session_id] = \
            get_feature(usr_inputs, dialog_acts, f_ngram_dict, fs_ngram_dict, fv_ngram_dict, slot_values_dict)

#=============load label============
    labels = {}
    for filename in filelist:
        filename = filename.strip()
        label_file = data_path + filename + '/label.json'

        session_id = get_session_id(label_file) 
        print (label_file)
        label_data = load_label(label_file)
    
        labels[session_id] = format_label(label_data, slot_values_dict);
    
 
    fw = open('b_features', 'wb')
    pickle.dump(input_features, fw)
    fw.close()
    
    fw = open('b_labels', 'wb')
    pickle.dump(labels, fw)
    fw.close()
