import numpy as np 
import pandas as pd 
import os
import csv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from speechpy import processing,feature
import scipy.io.wavfile as wav


def make_spec(file, file_dir, flip=False, ps=False, st = 4):
    """
    create a melspectrogram from the amplitude of the sound
    
    Args:
        file (str): filename
        file_dir (str): directory path
        flip (bool): reverse time axis
        ps (bool): pitch shift
        st (int): half-note steps for pitch shift
    Returns:
        np.array with shape (122,85) (time, freq)
    """
    
    sig, sr = librosa.load(file_dir+'/audio/'+file, sr=16000)
    
    if len(sig) < 16000: #pad shorter than 1 sec audio with ramp to zero
        sig = np.pad(sig, (0,16000-len(sig)), "linear_ramp")
        
    if ps:
        rate=16000
        sig = librosa.effects.pitch_shift(sig, rate, st)
        
    D = librosa.amplitude_to_db(librosa.stft(sig[:16000], 
                                             n_fft=512, 
                                             hop_length=128,
                                             center=False),
                               ref=np.max)
    S = librosa.feature.melspectrogram(S=D, n_mels=85).T
    
    if flip:
        S = np.flipud(S)
    
    return S.astype(np.float32)


def split_arr(arr):
    """
    split an array into chunks of length 16000
    Returns:
        list of arrays
    """
    return np.split(arr, np.arange(16000, len(arr), 16000))


def create_silence(train_dir):
    """
    reads wav files in background noises folder, 
    splits them and saves to silence folder in train_dir
    """
    for file in os.listdir(os.path.join(train_dir,"_background_noise_/")):
        if ".wav" in file:
            sig, sr = librosa.load(os.path.join(train_dir,"_background_noise_/")+file, sr = 16000) 
            sig_arr = split_arr(sig)
            if not os.path.exists(train_dir+"/audio/silence/"):
                os.makedirs(train_dir+"/audio/silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "frag%d" %ind + "_%s" %file # example: frag0_running_tap.wav
                sf.write(train_dir+"/audio/silence/"+file_name, arr, 16000)
                
def get_validation_list(train_dir):
    with open(train_dir+"/testing_list.txt") as val_list:
        validation_list = [row[0] for row in csv.reader(val_list)]
    assert len(validation_list) == 6835, "testing files not loaded"
    for i, file in enumerate(os.listdir(train_dir+"audio/silence/")):
        if i%10 == 0:
            validation_list.append("audio/silence/"+file)
    return validation_list

def get_test_val_lists(train_dir, validation_list):
    training_list  = []
    all_files_list = []
    class_counts = {}
    folders = os.listdir(train_dir+"/audio")
    for folder in folders:        
        files = os.listdir(os.path.join(train_dir,'audio',folder))
        for i, f in enumerate(files):
            all_files_list.append(folder+"/"+f)
            path = folder+'/'+f
            if path not in validation_list:
                training_list.append(folder+'/'+f)
            class_counts[folder] = i

    #remove filenames from validation_list that don't exist anymore (due to eda)
    validation_list = list(set(validation_list).intersection(all_files_list))
    return validation_list, training_list
def get_all_classes(train_dir):
    
    classes = os.listdir(os.path.join(train_dir,'audio'))
    
    if "_background_noise_" in classes:
        classes.remove("_background_noise_")
        
    folders = os.listdir(train_dir+"/audio")
    
    # put folders in same order as in the classes list, used when making sets
    all_classes = [x for x in classes]
    for ind, cl in enumerate(folders):
        if cl not in classes:
            all_classes.append(cl)
    return all_classes


def create_sets(file_list,all_classes,train_dir,method = 'spec'):
    if method=='spec':
        X_array = np.zeros([len(file_list), 122, 85])
    elif method=='fbank':
        X_array = np.zeros([len(file_list), 97, 80])
    y_array = np.zeros([len(file_list)])
    for ind, file in enumerate(file_list):
        if ind%2000 == 0:
            print(ind, file)
        if method == 'spec':
            X_array[ind] = make_spec(file,train_dir)
        elif method == 'fbank':
            X_array[ind] = wav_padding(compute_fbank(train_dir+'/audio/'+file),97, 80)
        else:
            raise ValueError("Invalid case")
        y_array[ind] = all_classes.index(file.rsplit('/')[0])
    return X_array, y_array


def compute_fbank(file):
    apply_cmvn = True
    sr, signal = wav.read(file)
    
    signal_preemphasized = processing.preemphasis(signal, cof=0.98)

    frames = processing.stack_frames(signal_preemphasized, sampling_frequency=sr,
                                     frame_length=0.025,
                                     frame_stride=0.010,
                                     zero_padding=True)


    power_spectrum = processing.power_spectrum(frames, fft_points=512) # num_frames x fft_length
    
    log_fbank = feature.lmfe(signal_preemphasized,sampling_frequency=sr,frame_length=0.025,
                        frame_stride=0.010,num_filters=80,
                         fft_length=512, low_frequency=0,high_frequency=None) # num_frames x num_filters


    if apply_cmvn:
        log_fbank_cmvn = processing.cmvn(log_fbank, variance_normalization=True)
        log_fbank = log_fbank_cmvn 


    return log_fbank

def wav_padding(wav_data, wav_max_len, feature_dim):    
    new_wav_data_lst = np.zeros(
        (wav_max_len, feature_dim),dtype=np.float32) # 如果增加了一阶和二阶导数则是三个channel，分别是static, first and second derivative features
    
    new_wav_data_lst[:wav_data.shape[0], :] = wav_data
    # print('new_wav_data_lst',new_wav_data_lst.shape,wav_lens.shape)
    return new_wav_data_lst