import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, help='Path to .wav file with respiratory')
args = parser.parse_args()

from wavfile import read
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from python_speech_features import mfcc
import pickle
import numpy as np

def read_wav(path):
    res = read(path)
    rate = res[0]
    sig = res[1]
    return rate, sig

def get_mfcc(sig, rate):
    mfcc_feat = mfcc(sig,rate,nfft=2300, lowfreq=50, highfreq=2000, winlen=0.05, winstep=0.05)
    return mfcc_feat

def get_features(path_to_wav, l=10):
    rate, signal = read_wav(path_to_wav)
    mfc = get_mfcc(signal, rate)
    samples = [mfc[j-l:j].flatten() for j in range(l, mfc.shape[0], l)]
    mms = pickle.load(open("./mms_akg.pkl", 'rb'))
    samples = mms.transform(samples)
    return np.array(samples)

def load_model():
    arch, weights = pickle.load(open("../model/4_model.pkl", 'rb'))
    model = model_from_json(arch)
    model.set_weights(weights)
    return model

def predict(path_to_wav):
    samples = get_features(path_to_wav)
    model = load_model()
    return model.predict(np.array([samples]))[1][0][0]

print("Probability of anomalies:", str(predict(args.wav)*100)[:4] + "%")