import librosa
import numpy as np
import data_tools as dt
import data_display as dd


def plot_out(temp_dir, sample_rate, hop_length_fft):

    input_file = dt.audio_files_to_file(temp_dir, 'Test Real - Raw.wav', sample_rate)
    after_file = dt.audio_files_to_file(temp_dir, 'Test Real - After.wav', sample_rate)
    noise_file = dt.audio_files_to_file(temp_dir, 'Test Real - Noise.wav', sample_rate)
    # clean_file = dt.audio_files_to_file(temp_dir, 'Test 2 - Clean.wav', sample_rate)
    output_file = dt.audio_files_to_file(temp_dir, 'Test Real - Final.wav', sample_rate)
    # dd.make_3plots_timeseries_voice_noise(input_file, clean_file, output_file, sample_rate)
    dd.make_2plots_timeseries_voice_after(input_file, after_file)
    dd.make_2plots_timeseries_voice_after(after_file, output_file)

    m_amp_db_input = librosa.amplitude_to_db(np.abs(librosa.stft(after_file)), ref=np.max)
    m_amp_db_noise = librosa.amplitude_to_db(np.abs(librosa.stft(noise_file)), ref=np.max)
    # m_amp_db_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_file)), ref=np.max)
    m_amp_db_output = librosa.amplitude_to_db(np.abs(librosa.stft(output_file)), ref=np.max)

    dd.make_3plots_spec_voice_noise(m_amp_db_input, m_amp_db_noise, m_amp_db_output, sample_rate, hop_length_fft)
    # dd.make_2plots_spec_clean_out(m_amp_db_output, m_amp_db_clean, sample_rate, hop_length_fft)
