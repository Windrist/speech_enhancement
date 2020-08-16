import librosa
import os
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import logmmse
import noisereduce as nr
import data_tools as dt
import data_display as dd


def prediction(weights_dir, model_name, input_dir, output_dir, sample_rate, frame_length, hop_length_frame, n_fft,
               hop_length_fft):

    json_file = open(weights_dir + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_dir + model_name + '.h5')
    print("Loaded model from disk!")

    list_files = os.listdir(input_dir)
    for file in list_files:
        logmmse.logmmse_from_file(input_file=input_dir + file, output_file=output_dir + 'Remastered-' + file)
        audio_temp = dt.audio_files_to_file(input_dir, file, sample_rate)
        # logmmse.logmmse(data=audio_temp, sampling_rate=32000, output_file=output_dir + 'Remastered-' + file)
        # audio_out = nr.reduce_noise(audio_clip=audio_temp, noise_clip=audio_temp, n_fft=n_fft+1, win_length=n_fft+1,
        #                             hop_length=hop_length_fft)
        # librosa.output.write_wav(output_dir + 'Remastered-' + file, audio_out, sample_rate)
        audio_file = dt.audio_files_to_file(output_dir, 'Remastered-' + file, sample_rate)

        # fig, ax = plt.subplots(figsize=(12, 6))
        # plt.title('Audio')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Time(s)')
        # ax.plot(audio_temp)
        # ax.plot(audio_file, alpha=0.5)
        # plt.show()

        audio_list = [audio_file]
        audio = dt.audio_list_to_numpy(audio_list, frame_length, hop_length_frame)

        dim_square_spec = int(n_fft / 2) + 1

        m_amp_db_audio, m_pha_audio = dt.audio_numpy_to_matrix_spectrogram(audio, dim_square_spec, n_fft,
                                                                           hop_length_fft)

        x_in = dt.scaled_in(m_amp_db_audio)
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[1], x_in.shape[2], 1)
        x_pred = loaded_model.predict(x_in)
        inv_sca_x_pred = dt.inv_scaled_out(x_pred)
        x_denoise = m_amp_db_audio - inv_sca_x_pred[:, :, :, 0]

        audio_denoise_recons = dt.matrix_spectrogram_to_numpy_audio(x_denoise, m_pha_audio, frame_length,
                                                                    hop_length_fft)
        nb_samples = audio_denoise_recons.shape[0]
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10
        librosa.output.write_wav(output_dir + 'Final-' + file, denoise_long[0, :], sample_rate)
        noise_recons = dt.matrix_spectrogram_to_numpy_audio(inv_sca_x_pred[:, :, :, 0], m_pha_audio, frame_length,
                                                            hop_length_fft)
        nb_samples = noise_recons.shape[0]
        noise_long = noise_recons.reshape(1, nb_samples * frame_length)
        librosa.output.write_wav(output_dir + 'Noise-' + file, noise_long[0, :], sample_rate)
