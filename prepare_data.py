import os

import librosa
import numpy as np
import data_tools as dt


def create_data(noise_dir, voice_dir, dataset_noise, dataset_voice, time_wave_dir, sound_dir, spectrogram_dir,
                sample_rate, min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft,
                hop_length_fft, list_noise_files, list_voice_files):

    noise_list = dt.audio_files_to_list(noise_dir + dataset_noise + '/', list_noise_files, sample_rate, min_duration)
    noise = dt.audio_list_to_numpy(noise_list, frame_length, hop_length_frame_noise)
    print("Noise Get Complete!")

    voice_list = dt.audio_files_to_list(voice_dir + dataset_voice + '/', list_voice_files, sample_rate, min_duration)
    voice = dt.audio_list_to_numpy(voice_list, frame_length, hop_length_frame)
    print("Voice Get Complete!")

    for i in range(1):
        prod_voice, prod_noise, prod_noisy_voice = dt.blend_noise_randomly(voice, noise, nb_samples, frame_length)
        print("Mix Sound Complete!")

        noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(sound_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_long.wav',
                                 noisy_voice_long[0, :], sample_rate)
        voice_long = prod_voice.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(sound_dir + str(i) + f'{dataset_voice}_voice_long.wav', voice_long[0, :], sample_rate)
        noise_long = prod_noise.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(sound_dir + str(i) + f'{dataset_noise}_noise_long.wav', noise_long[0, :], sample_rate)
        print("Combine Complete!")

        dim_square_spec = int(n_fft / 2) + 1

        m_amp_db_voice, m_pha_voice = dt.audio_numpy_to_matrix_spectrogram(prod_voice, dim_square_spec, n_fft,
                                                                           hop_length_fft)
        m_amp_db_noise, m_pha_noise = dt.audio_numpy_to_matrix_spectrogram(prod_noise, dim_square_spec, n_fft,
                                                                           hop_length_fft)
        m_amp_db_noisy_voice, m_pha_noisy_voice = dt.audio_numpy_to_matrix_spectrogram(prod_noisy_voice, dim_square_spec,
                                                                                       n_fft, hop_length_fft)
        print("Create Amplitude Complete!")

        np.save(time_wave_dir + str(i) + f'{dataset_voice}_voice_time_wave', prod_voice)
        np.save(time_wave_dir + str(i) + f'{dataset_noise}_noise_time_wave', prod_noise)
        np.save(time_wave_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_time_wave', prod_noisy_voice)

        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_voice_amp_db', m_amp_db_voice)
        np.save(spectrogram_dir + str(i) + f'{dataset_noise}_noise_amp_db', m_amp_db_noise)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_amp_db', m_amp_db_noisy_voice)

        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_voice_pha_db', m_pha_voice)
        np.save(spectrogram_dir + str(i) + f'{dataset_noise}_noise_pha_db', m_pha_noise)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_pha_db', m_pha_noisy_voice)
        print("Training Set Complete!")


def create_data_special(noise_dir, voice_dir, dataset_noise, dataset_voice, time_wave_dir, sound_dir, spectrogram_dir,
                        sample_rate, min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples,
                        n_fft, hop_length_fft, list_noise_files, list_voice_files):

    noise_list = dt.audio_files_to_list(noise_dir + dataset_noise + '/', list_noise_files, sample_rate, min_duration)
    noise = dt.audio_list_to_numpy(noise_list, frame_length, hop_length_frame_noise)
    print("Mix Sound Get Complete!")

    voice_list = dt.audio_files_to_list(voice_dir + dataset_voice + '/', list_voice_files, sample_rate, min_duration)
    voice = dt.audio_list_to_numpy(voice_list, frame_length, hop_length_frame)
    print("Voice Get Complete!")

    for i in range(1):
        prod_voice, prod_noise = dt.blend_no_need_noise(voice, noise, nb_samples, frame_length)

        voice_long = prod_voice.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(sound_dir + str(i) + f'{dataset_voice}_voice_long.wav', voice_long[0, :], sample_rate)
        noise_long = prod_noise.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(sound_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_long.wav', noise_long[0, :], sample_rate)
        print("Combine Complete!")

        dim_square_spec = int(n_fft / 2) + 1

        m_amp_db_voice, m_pha_voice = dt.audio_numpy_to_matrix_spectrogram(prod_voice, dim_square_spec, n_fft,
                                                                           hop_length_fft)
        m_amp_db_noise, m_pha_noise = dt.audio_numpy_to_matrix_spectrogram(prod_noise, dim_square_spec, n_fft,
                                                                           hop_length_fft)
        print("Create Amplitude Complete!")

        np.save(time_wave_dir + str(i) + f'{dataset_voice}_voice_time_wave', prod_voice)
        np.save(time_wave_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_time_wave', prod_noise)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_voice_amp_db', m_amp_db_voice)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_amp_db', m_amp_db_noise)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_voice_pha_db', m_pha_voice)
        np.save(spectrogram_dir + str(i) + f'{dataset_voice}_{dataset_noise}_noisy_voice_pha_db', m_pha_noise)
        print("Training Set Complete!")
