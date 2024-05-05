import numpy as np 
import os
import csv
import shutil
import librosa
import librosa.display
import soundfile as sf
from speechpy import processing,feature
import scipy.io.wavfile as wav

FREQ = 16000
PERC = 10

# generates and saves preprocessed data files
def gen_files(train_dir, method = 'spec', folder_name = 'data'):
    if not os.path.exists('data'):
        os.mkdir('data')

    background_noise_dir = os.path.join(train_dir, "audio", "_background_noise_")

    if os.path.exists(background_noise_dir):
        shutil.move(background_noise_dir, train_dir)

    create_silence(train_dir)
    classes = get_all_classes(train_dir)

    validation_list = get_validation_list(train_dir)
    training_list = get_test_val_lists(train_dir,validation_list)

    X_train, y_train = create_sets(training_list, classes, train_dir, method = method)
    X_val, y_val = create_sets(validation_list, classes, train_dir, method = method)

    np.save(f"{folder_name}/X_train.npy", np.expand_dims(X_train, -1)+1.3)
    np.save(f"{folder_name}/y_train.npy", y_train.astype(int))
    np.save(f"{folder_name}/X_val.npy", np.expand_dims(X_val, -1)+1.3)
    np.save(f"{folder_name}/y_val.npy", y_val.astype(int))

def make_spec(file, file_dir, flip=False, ps=False, st = 4):
    sig, _ = librosa.load(file_dir+'/audio/'+file, sr=FREQ)
    
    if len(sig) < FREQ: 
        sig = np.pad(sig, (0, FREQ-len(sig)), "linear_ramp")
        
    if ps:
        rate=FREQ
        sig = librosa.effects.pitch_shift(sig, rate, st)
        
    D = librosa.amplitude_to_db(librosa.stft(sig[:FREQ], n_fft=512, hop_length=128, center=False),
                                ref=np.max)
    S = librosa.feature.melspectrogram(S=D, n_mels=85).T
    
    if flip:
        S = np.flipud(S)
    
    return S.astype(np.float32)

def create_silence(train_dir):
    for file in os.listdir(os.path.join(train_dir,"_background_noise_/")):
        if ".wav" in file:
            sig, _ = librosa.load(os.path.join(train_dir,"_background_noise_/")+file, sr = FREQ) 
            sig_arr = np.split(sig, np.arange(FREQ, len(sig), FREQ))
            if not os.path.exists(train_dir+"/audio/silence/"):
                os.makedirs(train_dir+"/audio/silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "frag%d" %ind + "_%s" %file
                sf.write(train_dir+"/audio/silence/"+file_name, arr, FREQ)
                
def get_validation_list(train_dir):
    with open(train_dir+"/validation_list.txt") as val_list:
        validation_list = [row[0] for row in csv.reader(val_list)]

    for i, file in enumerate(os.listdir(train_dir+"/audio/silence/")):
        if i % PERC== 0:
            validation_list.append("silence/"+file)
    return validation_list

def get_test_val_lists(train_dir, validation_list):
    training_list  = []
    class_counts = {}
    folders = os.listdir(train_dir+"/audio")
    for folder in folders:        
        files = os.listdir(os.path.join(train_dir,'audio',folder))
        for i, f in enumerate(files):
            path = folder+'/'+f
            if path not in validation_list:
                training_list.append(folder+'/'+f)
            class_counts[folder] = i

    return training_list

def get_all_classes(train_dir):
    classes = os.listdir(os.path.join(train_dir,'audio'))
    
    if "_background_noise_" in classes:
        classes.remove("_background_noise_")
        
    folders = os.listdir(train_dir+"/audio")
    
    all_classes = [x for x in classes]
    for ind, cl in enumerate(folders):
        if cl not in classes:
            all_classes.append(cl)
    return all_classes


def create_sets(file_list, all_classes, train_dir, method = 'spec'):
    if method=='spec':
        X_array = np.zeros([len(file_list), 122, 85])
    elif method=='fbank':
        X_array = np.zeros([len(file_list), 97, 80])
    y_array = np.zeros([len(file_list)])
    for ind, file in enumerate(file_list):

        if method == 'spec':
            X_array[ind] = make_spec(file,train_dir)
        elif method == 'fbank':
            X_array[ind] = wav_padding(compute_fbank(train_dir+'/audio/'+file), 97, 80)
        else:
            raise ValueError("Invalid case")
        y_array[ind] = all_classes.index(file.rsplit('/')[0])
    return X_array, y_array


def compute_fbank(file):
    apply_cmvn = True
    sr, signal = wav.read(file)
    
    signal_preemphasized = processing.preemphasis(signal, cof=0.98)
    
    log_fbank = feature.lmfe(signal_preemphasized,
                             sampling_frequency=sr,
                             frame_length=0.025,
                             frame_stride=0.010,
                             num_filters=80,
                             fft_length=512, 
                             low_frequency=0,
                             high_frequency=None) 

    if apply_cmvn:
        log_fbank_cmvn = processing.cmvn(log_fbank, variance_normalization=True)
        log_fbank = log_fbank_cmvn 

    return log_fbank

def wav_padding(wav_data, wav_max_len, feature_dim):    
    new_wav_data_lst = np.zeros(
        (wav_max_len, feature_dim),dtype=np.float32)
    
    new_wav_data_lst[:wav_data.shape[0], :] = wav_data

    return new_wav_data_lst

