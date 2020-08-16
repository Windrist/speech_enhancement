from __future__ import division
import math
import numpy as np
from scipy.special import expn


def base(x, srate, noise_frames=6, slen=0, eta=0.15, saved_params=None):
    if slen == 0:
        slen = int(math.floor(0.02 * srate))

    if slen % 2 == 1:
        slen = slen + 1

    perc = 50
    len1 = int(math.floor(slen * perc / 100))
    len2 = int(slen - len1)

    win = np.hanning(slen)
    win = win * len2 / np.sum(win)
    nfft = 2 * slen

    x_old = np.zeros(len1)
    xk_prev = np.zeros(len1)
    nframes = int(math.floor(x.size / len2) - math.floor(slen / len2))
    xfinal = np.zeros(nframes * len2)

    if saved_params is None:
        noise_mean = np.zeros(nfft)
        for j in range(0, slen * noise_frames, slen):
            noise_mean = noise_mean + np.absolute(np.fft.fft(win * x[0, j:j + slen], nfft, axis=0))
        noise_mu2 = (noise_mean / noise_frames) ** 2
    else:
        noise_mu2 = saved_params['noise_mu2']
        xk_prev = saved_params['xk_prev']
        x_old = saved_params['x_old']

    aa = 0.98
    mu = 0.98
    eta = 0.15
    ksi_min = 10 ** (-25 / 10)

    for k in range(0, nframes * len2, len2):
        insign = win * x[0, k:k + slen]

        spec = np.fft.fft(insign, nfft, axis=0)
        sig = np.absolute(spec)
        sig2 = sig ** 2

        gammak = np.minimum(sig2 / noise_mu2, 40)

        if xk_prev.all() == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(ksi_min, ksi)

        log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        vad_decision = np.sum(log_sigma_k) / slen
        if vad_decision < eta:
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2

        a = ksi / (1 + ksi)
        vk = a * gammak
        ei_vk = 0.5 * expn(1, vk)
        hw = a * np.exp(ei_vk)
        sig = sig * hw
        xk_prev = sig ** 2
        xi_w = np.fft.ifft(hw * spec, nfft, axis=0)
        xi_w = np.real(xi_w)

        xfinal[k:k + len2] = x_old + xi_w[0:len1]
        x_old = xi_w[len1:slen]

    return xfinal, {'noise_mu2': noise_mu2, 'xk_prev': xk_prev, 'x_old': x_old}


def mono_logmmse(m_input, fs, dtype, initial_noise, window_size, noise_threshold):
    num_frames = len(m_input)
    chunk_size = int(np.floor(60*fs))
    m_output = np.array([], dtype=dtype)
    saved_params = None
    frames_read = 0
    while frames_read < num_frames:
        frames = num_frames - frames_read if frames_read + chunk_size > num_frames else chunk_size
        signal = m_input[frames_read:frames_read + frames]
        frames_read = frames_read + frames
        _output, saved_params = base(signal, fs, initial_noise, window_size, noise_threshold, saved_params)
        m_output = np.concatenate((m_output, from_float(_output, dtype)))
    return m_output


def to_float(_input):
    if _input.dtype == np.float64:
        return _input, _input.dtype
    elif _input.dtype == np.float32:
        return _input.astype(np.float64), _input.dtype
    elif _input.dtype == np.uint8:
        return (_input - 128) / 128., _input.dtype
    elif _input.dtype == np.int16:
        return _input / 32768., _input.dtype
    elif _input.dtype == np.int32:
        return _input / 2147483648., _input.dtype
    raise ValueError('Unsupported wave file format, please contact the author')


def from_float(_input, dtype):
    if dtype == np.float64:
        return _input, np.float64
    elif dtype == np.float32:
        return _input.astype(np.float32)
    elif dtype == np.uint8:
        return ((_input * 128) + 128).astype(np.uint8)
    elif dtype == np.int16:
        return (_input * 32768).astype(np.int16)
    elif dtype == np.int32:
        print(_input)
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format, please contact the author')


def logmmse(data, sampling_rate, initial_noise=6, window_size=0, noise_threshold=0.15):
    data = np.array(data)
    data, dtype = to_float(data)
    data += np.finfo(np.float64).eps
    output = mono_logmmse(data, sampling_rate, dtype, initial_noise, window_size, noise_threshold)
    output = np.array(output)
    return output.T
