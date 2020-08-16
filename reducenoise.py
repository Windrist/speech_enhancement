import scipy.signal
import numpy as np
import os
import librosa
import tensorflow as tf


def _stft(y, n_fft, hop_length, win_length, use_tensorflow=False):
    if use_tensorflow:
        return _stft_tensorflow(y, n_fft, hop_length, win_length)
    else:
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)


def _istft(y, n_fft, hop_length, win_length, use_tensorflow=False):
    if use_tensorflow:
        return _istft_tensorflow(y.T, n_fft, hop_length, win_length)
    else:
        return librosa.istft(y, hop_length, win_length)


def _stft_librosa(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)


def _istft_librosa(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _stft_tensorflow(y, n_fft, hop_length, win_length):
    return tf.signal.stft(y, win_length, hop_length, n_fft, pad_end=True, window_fn=tf.signal.hann_window).numpy().T


def _istft_tensorflow(y, n_fft, hop_length, win_length):
    return tf.signal.inverse_stft(y.astype(np.complex64), win_length, hop_length, n_fft).numpy()


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x, ):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def _smoothing_filter(n_grad_freq, n_grad_time):

    smoothing_filter = np.outer(np.concatenate([np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                                                np.linspace(1, 0, n_grad_freq + 2)])[1:-1],
                                np.concatenate([np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                                                np.linspace(1, 0, n_grad_time + 2)])[1:-1])
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    return smoothing_filter


def mask_signal(sig_stft, sig_mask):
    sig_stft_amp = sig_stft * (1 - sig_mask)
    return sig_stft_amp


def convolve_gaussian(sig_mask, smoothing_filter, use_tensorflow=False):
    if use_tensorflow:
        smoothing_filter = smoothing_filter * ((np.shape(smoothing_filter)[1] - 1) / 2 + 1)
        smoothing_filter = smoothing_filter[:, :, tf.newaxis, tf.newaxis].astype()
        img = sig_mask[:, :, tf.newaxis, tf.newaxis].astype("float32")
        return tf.nn.conv2d(img, smoothing_filter, strides=[1, 1, 1, 1], padding="SAME").numpy().squeeze()
    else:
        return scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")


def reduce_noise(input_dir, temp_dir, n_grad_freq=2, n_grad_time=4, n_fft=2048, win_length=2048, hop_length=512,
                 n_std_thresh=1.5, prop_decrease=1.0, pad_clipping=True, use_tensorflow=False):

    list_input_files = os.listdir(input_dir)
    for file in list_input_files:
        noise_clip, sr = librosa.load(os.path.join(input_dir, file), sr=16000)
        noise_stft = _stft(noise_clip, n_fft, hop_length, win_length, use_tensorflow=use_tensorflow)
        noise_stft_db = _amp_to_db(np.abs(noise_stft))

        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

        if pad_clipping:
            nsamp = len(noise_clip)
            noise_clip = np.pad(noise_clip, [0, hop_length], mode="constant")

        sig_stft = _stft(noise_clip, n_fft, hop_length, win_length, use_tensorflow=use_tensorflow)
        sig_stft_db = _amp_to_db(np.abs(sig_stft))

        db_thresh = np.repeat(np.reshape(noise_thresh, [1, len(mean_freq_noise)]), np.shape(sig_stft_db)[1], axis=0).T
        sig_mask = sig_stft_db < db_thresh
        smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)
        sig_mask = convolve_gaussian(sig_mask, smoothing_filter, use_tensorflow)
        sig_mask = sig_mask * prop_decrease
        sig_stft_amp = mask_signal(sig_stft, sig_mask)

        recovered_signal = _istft(sig_stft_amp, n_fft, hop_length, win_length, use_tensorflow=use_tensorflow)
        if pad_clipping:
            recovered_signal = librosa.util.fix_length(recovered_signal, nsamp)

        librosa.output.write_wav(temp_dir + file, recovered_signal, 16000)
