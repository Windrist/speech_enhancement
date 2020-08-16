import librosa
import numpy as np
import os


def audio_numpy_to_frame_stack(sound_data, frame_length, hop_length_frame):

    sequence_sample_length = sound_data.shape[0]
    sound_data_list = []
    sequence_sample_length_new = sequence_sample_length + (frame_length - (sequence_sample_length % frame_length)) + 1
    sound_data = np.append(sound_data, np.zeros(frame_length - (sequence_sample_length % frame_length) + 1))

    for start in range(0, sequence_sample_length_new - frame_length + 1, hop_length_frame):
        sound_data_list.append(sound_data[start:start + frame_length])

    return np.vstack(sound_data_list)


def audio_list_to_numpy(list_sound_array, frame_length, hop_length_frame):

    list_num_array = []

    for sound in list_sound_array:
        list_num_array.append(audio_numpy_to_frame_stack(sound, frame_length, hop_length_frame))

    return np.vstack(list_num_array)


def audio_files_to_list(audio_dir, list_audio_files, sample_rate, min_duration):

    list_sound_array = []

    for file in list_audio_files:
        y = audio_files_to_file(audio_dir, file, sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sample_rate)

        if total_duration >= min_duration:
            list_sound_array.append(y)
            print(f"Import: {file} Successful!")
        else:
            print(f"The following file {os.path.join(audio_dir, file)} is below the min duration")

    return list_sound_array


def audio_files_to_file(audio_dir, file, sample_rate):
    y, sr = librosa.load(audio_dir + file, sr=sample_rate)
    # if sr != sample_rate:
    #     data_out = librosa.core.resample(y, sr, sample_rate, fix=False, scale=True)
    return y


def blend_noise_randomly(voice, noise, nb_samples, frame_length):

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice


def blend_no_need_noise(voice, noise, nb_samples, frame_length):

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = noise[id_noise, :]

    return prod_voice, prod_noise


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):

    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stft_audio_magnitude, stft_audio_phase = librosa.magphase(stft_audio)

    stft_audio_magnitude_db = librosa.amplitude_to_db(
        stft_audio_magnitude, ref=np.max)

    return stft_audio_magnitude_db, stft_audio_phase


def audio_numpy_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):

    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase


def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stft_audio_magnitude_db, stft_audio_phase):

    stft_audio_magnitude_rev = librosa.db_to_amplitude(stft_audio_magnitude_db, ref=1.0)

    # taking magnitude and phase of audio
    audio_reverse_stft = stft_audio_magnitude_rev * stft_audio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct


def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft):

    nb_spec = m_mag_db.shape[0]
    list_audio = []

    for i in range(nb_spec):
        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)


def scaled_in(matrix_spec):
    matrix_spec = (matrix_spec + 46) / 50
    return matrix_spec


def scaled_out(matrix_spec):
    matrix_spec = (matrix_spec - 6) / 82
    return matrix_spec


def inv_scaled_in(matrix_spec):
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec


def inv_scaled_out(matrix_spec):
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec
