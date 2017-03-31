# dstc_rnn
a dynamic RNN model of DSTC3 task implemented in Tensorflow.

It's a implemention of the slot-dependent model mentioned in the paper "ROBUST DIALOG STATE TRACKING USING DELEXICALISED RECURRENT NEURAL NETWORKS AND UNSUPERVISED ADAPTATION".


## WARNING
The model in the paper is consist of a slot-independent model and a number of slot-dependent models and the slot-independent model is used to initialize the slot-dependent models.

Until now, however, I have just done the slot-dependent models. In addition, the model I did is not exactly the same as the auther did and I am not sure if I did the feature engineering correctly. 

So it's not a surprise that the result is bad.

## Requirements
Python 3.5.2
Tensorflow >= 1.1.0

## Dataset
You can find the data from this link: http://camdial.org/~mh521/dstc/

## Usage
### 1. Get features and labels from the raw data. 
The output file is 'b_features' and 'b_labels'. The path of data needs to be modified in the code.
```
python preprocess.py
```

### 2. run the model.
```
python dynamic_rnn.py [slot_name] [train_iters] [learning_rate] [batch_size] [model_name]
```
for example:
```
python3 dynamic_rnn.py pricerange 12800 0.01 128 test
```
## Reference
1. Paper link: http://svr-www.eng.cam.ac.uk/~sjy/papers/htyo14.pdf
2. https://github.com/aymericdamien/TensorFlow-Examples/
