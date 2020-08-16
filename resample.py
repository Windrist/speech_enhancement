import os

import librosa

list_files = os.listdir('E:/Research/Sound Recognition/Dataset/Noise_dataset/Metal/')

for file in list_files:
    data, samplerate = librosa.load(os.path.join('E:/Research/Sound Recognition/Dataset/Noise_dataset/Metal/', file))
    print(f"Import: {file} Successful!")
    data_out = librosa.core.resample(data, 24000, 16000, scale=True)
    librosa.output.write_wav('E:/Research/Sound Recognition/Dataset/Noise_dataset/Metal_vert/' + file, data_out, 16000)
    print(f"Convert: {file} Successful!")
