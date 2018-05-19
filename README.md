# Lung Diagnostics
This repository is an implementation of Noise Masking RNN for respiratory sound classification proposed in our [paper](paper.pdf)

![alt text](https://raw.githubusercontent.com/kirtov/lung_diagnostics/master/rnn_schema.png)


It includes a preprocessing functions to convert raw .wav respiratory sound files and train script

# For training
Download model and data dirs from [here](https://drive.google.com/open?id=1HITRmN5YCErIUrxOjTZE-vZmpykTB48x)
```
Run python3 train.py --gpu GPU_NUM --data_path DATA_PATH --cv_path CV_SPLIT_PATH --exp_path EXPERIMENT_PATH
```
(data and cv split are included [here](https://drive.google.com/open?id=1HITRmN5YCErIUrxOjTZE-vZmpykTB48x) in /data directory)

# For testing

```
Run python3 predict.py --wav WAVFILE_PATH
```
Script prints text: "Probability of anomalies: X%"
