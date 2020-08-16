import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_unet import unet
from data_tools import scaled_in, scaled_out


def training(data_noise_dir, data_voice_dir, spectrogram_dir, weights_dir, model_name, training_from_scratch, epochs, batch_size):

    for i in range(10):
        if i == 0:
            training_from_scratch = True
        else:
            training_from_scratch = False
        x_in = np.load(spectrogram_dir + str(i) + f'{data_voice_dir}_{data_noise_dir}_noisy_voice_amp_db' + ".npy")
        x_out = np.load(spectrogram_dir + str(i) + f'{data_voice_dir}_voice_amp_db' + ".npy")

        x_out = x_in - x_out

        print(stats.describe(x_in.reshape(-1, 1)))
        print(stats.describe(x_out.reshape(-1, 1)))

        x_in = scaled_in(x_in)
        x_out = scaled_out(x_out)

        print(x_in.shape)
        print(x_out.shape)

        print(stats.describe(x_in.reshape(-1, 1)))
        print(stats.describe(x_out.reshape(-1, 1)))

        x_in = x_in[:, :, :]
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[1], x_in.shape[2], 1)
        x_out = x_out[:, :, :]
        x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], 1)

        x_train, x_test, y_train, y_test = train_test_split(x_in, x_out, test_size=0.10, random_state=42)

        if training_from_scratch:

            generator_nn = unet()
        else:

            generator_nn = unet(pretrained_weights=weights_dir + model_name + '.h5')

        checkpoint = ModelCheckpoint(weights_dir + model_name + '.h5', verbose=1, monitor='val_loss', save_best_only=True,
                                     mode='auto')

        generator_nn.summary()
        history = generator_nn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                   callbacks=[checkpoint], verbose=1, validation_data=(x_test, y_test))
        model_json = generator_nn.to_json()
        with open(f"{weights_dir + model_name}.json", "w") as json_file:
            json_file.write(model_json)
