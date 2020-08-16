import argparse

parser = argparse.ArgumentParser(description='Speech enhancement, training and prediction')

# Mode to run the program (Options: creat, train or predict)
parser.add_argument('--mode', default='train', type=str, choices=['train', 'predict', 'plot'])

# Dataset name for training
parser.add_argument('--dataset_noise', default='Wind', type=str)
parser.add_argument('--dataset_voice', default='Water', type=str)

# Folders where to find noise audios and clean voice audio to prepare training dataset (mode: create)
parser.add_argument('--noise_dir',
                    default='/media/windrist/hdd2/Research/Main/Sound_ws/Dataset/Noise/',
                    type=str)
parser.add_argument('--voice_dir',
                    default='/media/windrist/hdd2/Research/Main/Sound_ws/Dataset/Voice/',
                    type=str)

# Folders where to save spectrogram, time series and sounds for training / QC
parser.add_argument('--spectrogram_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Train/Spectrogram/',
                    type=str)

parser.add_argument('--time_wave_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Train/Time_wave/',
                    type=str)

parser.add_argument('--sound_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Train/Mix_sound/',
                    type=str)

# How much frame to create (mode: create)
parser.add_argument('--nb_samples', default=3600, type=int)

# Training from scratch or pre-trained weights
parser.add_argument('--training_from_scratch', default=True, type=bool)

# Folder of saved weights
parser.add_argument('--weights_dir', default='/home/windrist/Workspace/Sound_ws/src/Data/Weights/',
                    type=str)

# Nb of epochs for training
parser.add_argument('--epochs', default=20, type=int)

# Batch size for training
parser.add_argument('--batch_size', default=20, type=int)

# Name of saved model to read
parser.add_argument('--model_name', default='model_test', type=str)

# Directory where read noisy sound to denoise (mode: predict)
parser.add_argument('--input_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Result/Input/', type=str)

# Directory to save the denoise sound (mode: predict)
parser.add_argument('--output_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Result/Output/', type=str)
parser.add_argument('--temp_dir',
                    default='/home/windrist/Workspace/Sound_ws/src/Data/Result/Final/Test Ver2/',
                    type=str)

# Sample rate chosen to read audio
parser.add_argument('--sample_rate_train', default=8000, type=int)
parser.add_argument('--sample_rate_predict', default=8000, type=int)
parser.add_argument('--sample_rate_plot', default=4000, type=int)

# Minimum duration of audio files to consider
parser.add_argument('--min_duration', default=1.0, type=float)

# Training data will be frame of slightly above 1 second
parser.add_argument('--frame_length', default=8064, type=int)

# Hop length for clean voice files separation (no overlap)
parser.add_argument('--hop_length_frame', default=8064, type=int)

# Hop length for noise files to blend (noise is separated into several windows)
parser.add_argument('--hop_length_frame_noise', default=8064, type=int)

# Choosing n_fft and hop_length_fft to have squared spectrograms
parser.add_argument('--n_fft', default=255, type=int)
parser.add_argument('--hop_length_fft', default=63, type=int)
