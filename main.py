from args import parser
from prepare_data import create_data, create_data_special
from train_model import training
from prediction_denoise import prediction
from plot import plot_out
import os

if __name__ == '__main__':

    args = parser.parse_args()

    mode = args.mode
    training_mode = False
    prediction_mode = False
    plot_mode = False

    if mode == 'predict':
        prediction_mode = True
    elif mode == 'train':
        training_mode = True
    elif mode == 'plot':
        plot_mode = True

    if training_mode:
        noise_dir = args.noise_dir
        voice_dir = args.voice_dir
        dataset_noise = args.dataset_noise
        dataset_voice = args.dataset_voice
        time_wave_dir = args.time_wave_dir
        sound_dir = args.sound_dir
        spectrogram_dir = args.spectrogram_dir
        sample_rate = args.sample_rate_train
        min_duration = args.min_duration
        frame_length = args.frame_length
        hop_length_frame = args.hop_length_frame
        hop_length_frame_noise = args.hop_length_frame_noise
        nb_samples = args.nb_samples
        n_fft = args.n_fft
        hop_length_fft = args.hop_length_fft
        weights_dir = args.weights_dir
        model_name = args.model_name
        training_from_scratch = args.training_from_scratch
        epochs = args.epochs
        batch_size = args.batch_size

        list_noise_files = os.listdir(noise_dir + dataset_noise + '/')
        list_voice_files = os.listdir(voice_dir + dataset_voice + '/')

        if dataset_noise == 'Metal':
            print('Special Confirmed!')
            create_data_special(noise_dir, voice_dir, dataset_noise, dataset_voice, time_wave_dir, sound_dir,
                                spectrogram_dir, sample_rate, min_duration, frame_length, hop_length_frame,
                                hop_length_frame_noise, nb_samples, n_fft, hop_length_fft, list_noise_files,
                                list_voice_files)
            training(dataset_noise, dataset_voice, spectrogram_dir, weights_dir, model_name,
                     training_from_scratch, epochs, batch_size)
        else:
            # create_data(noise_dir, voice_dir, dataset_noise, dataset_voice, time_wave_dir, sound_dir,
            #             spectrogram_dir, sample_rate, min_duration, frame_length, hop_length_frame,
            #             hop_length_frame_noise, nb_samples, n_fft, hop_length_fft, list_noise_files,
            #             list_voice_files)
            training(dataset_noise, dataset_voice, spectrogram_dir, weights_dir, model_name,
                     training_from_scratch, epochs, batch_size)

    elif prediction_mode:

        weights_dir = args.weights_dir
        model_name = args.model_name
        input_dir = args.input_dir
        output_dir = args.output_dir
        sample_rate = args.sample_rate_predict
        min_duration = args.min_duration
        frame_length = args.frame_length
        hop_length_frame = args.hop_length_frame
        n_fft = args.n_fft
        hop_length_fft = args.hop_length_fft

        prediction(weights_dir, model_name, input_dir, output_dir, sample_rate, frame_length, hop_length_frame, n_fft,
                   hop_length_fft)

    elif plot_mode:

        temp_dir = args.temp_dir
        sample_rate = args.sample_rate_plot
        hop_length_fft = args.hop_length_fft

        plot_out(temp_dir, sample_rate, hop_length_fft)