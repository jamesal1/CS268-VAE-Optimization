import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

CONFIG = tf.ConfigProto(device_count={'GPU': 4}, log_device_placement=False, allow_soft_placement=False)
CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Layer
from keras.models import Model
from keras.datasets import mnist
from keras import losses
import numpy as np
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from numpy import genfromtxt
from numpy import linalg as LA

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# returns just one digit
def digitMatrix(x, y, digit):
    x_digit = []
    y_digit = []
    for i in range(len(y)):
        if y[i] == digit:
            x_digit.append(x[i])
            y_digit.append(y[i])
    return x_digit, y_digit


x_train_0, y_train_0 = digitMatrix(x_train, y_train, 0)
x_test_0, y_test_0 = digitMatrix(x_test, y_test, 0)
y_train_0_cat = keras.utils.to_categorical(y_train_0, num_classes)
y_test_0_cat = keras.utils.to_categorical(y_test_0, num_classes)
x_train_0 = np.reshape(x_train_0, (len(x_train_0), 28, 28, 1))
x_test_0 = np.reshape(x_test_0, (len(x_test_0), 28, 28, 1))

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# add noisy data for bad activations
noise_level = 0.5
noise = np.multiply(np.random.rand(len(x_train), 28, 28, 1), noise_level)
noisy_x_train = np.add(noise, x_train)
x_train = np.append(x_train, noisy_x_train, axis=0)
noise = np.multiply(np.random.rand(len(x_test), 28, 28, 1), noise_level)
noisy_x_test = np.add(noise, x_test)
x_test = np.append(x_test, noisy_x_test, axis=0)
y_train_cat = np.append(y_train_cat, y_train_cat, axis=0)
y_test_cat = np.append(y_test_cat, y_test_cat, axis=0)
y_test = np.append(y_test, y_test, axis=0)
y_train = np.append(y_train, y_train, axis=0)


def CreateModel(gpu=3):
    input_shape = (28, 28, 1)

    with tf.device('/gpu:' + str(gpu)):
        # convolutional autoencoder
        input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        print('shape of encoded', K.int_shape(encoded))

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        print('shape of decoded', K.int_shape(decoded))

        autoencoder = Model(input_img, decoded)
        autoencoder.load_weights("checkpoints/autoencoder_chkpt_epochs1000.hdf5")
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # mnist cnn
        cnn = Sequential()
        cnn.add(Conv2D(32, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=input_shape))
        cnn.add(Conv2D(64, (3, 3), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())
        cnn.add(Dense(128, activation='relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(num_classes))
        cnn.add(Activation("softmax"))

        cnn.load_weights("checkpoints/mnist_cnn_chkpt_epochs50.hdf5")
        cnn.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        # autoencoder plus CNN (CNN has frozen weights)

        inp = Input(shape=input_shape)
        h = Conv2D(16, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[1].get_weights(),
                   input_shape=input_shape)(inp)
        h = MaxPooling2D((2, 2), padding='same', weights=autoencoder.layers[2].get_weights())(h)
        h = Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[3].get_weights())(h)
        h = MaxPooling2D((2, 2), padding='same', weights=autoencoder.layers[4].get_weights())(h)
        h = Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[5].get_weights())(h)
        encoded = MaxPooling2D((2, 2), padding='same', name="blah", weights=autoencoder.layers[6].get_weights())(h)
        h = Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[7].get_weights(),
                   input_shape=(4, 4, 8))(encoded)
        h = UpSampling2D((2, 2), weights=autoencoder.layers[8].get_weights())(h)
        h = Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[9].get_weights())(h)
        h = UpSampling2D((2, 2), weights=autoencoder.layers[10].get_weights())(h)
        h = Conv2D(16, (3, 3), activation='relu', weights=autoencoder.layers[11].get_weights())(h)
        h = UpSampling2D((2, 2), weights=autoencoder.layers[12].get_weights())(h)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', weights=autoencoder.layers[13].get_weights())(
            h)
        h = Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   weights=cnn.layers[0].get_weights(), trainable=False, input_shape=input_shape)(decoded)
        h = Conv2D(64, (3, 3), activation='relu', weights=cnn.layers[1].get_weights(), trainable=False)(h)
        h = MaxPooling2D(pool_size=(2, 2), weights=cnn.layers[2].get_weights(), trainable=False)(h)
        h = Dropout(0.25, weights=cnn.layers[3].get_weights(), trainable=False)(h)
        h = Flatten()(h)
        h = Dense(128, activation='relu', weights=cnn.layers[5].get_weights(), trainable=False)(h)
        h = Dropout(0.5, weights=cnn.layers[6].get_weights(), trainable=False)(h)
        h = Dense(num_classes, weights = cnn.layers[7].get_weights(), trainable=False)(h)
        out = Activation("softmax") (h)

        model = Model(inp, [decoded, out])
        # model = Model(inp, out)
        model.load_weights("checkpoints/reg_autoencoder_chkptall_digs_2_dims_noise1_6_epochs250_alpha1.hdf5")
        # model.load_weights("checkpoints/reg_autoencoder_chkpttestnew_epochs50_alpha1.hdf5")

    return model, cnn


def train_model(model, epochs, index, normalization=100, batch_size=128, gpu=0, alpha=16, lr=0.001):
    checkpoint_path = 'checkpoints/reg_autoencoder_chkpt' + str(index) + '_epochs' + str(epochs) + '_alpha' + str(
        alpha) + '.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path)
    callbacks_list = [checkpoint]

    def custom_loss(y_true, y_pred):
        encoded = tf.reshape(model.layers[6].output, [-1, 128])
        # norms = tf.norm(encoded, axis=1, keep_dims=True)
        norms = tf.norm(encoded[:, 0:2], axis=1, keep_dims=True)
        digit_activations = tf.reduce_sum(tf.multiply(y_true, y_pred), 1, keep_dims=True)

        # categorical to continuous
        # need to rewrite, just vector nx1 of .5
        tensorlist = []
        for i in range(10):
            # tensorlist.append(tf.scalar_mul(i/10.0+1.0, y_true[:,i:i+1]))
            # tensorlist.append(tf.scalar_mul(i/20.0+0.5, y_true[:,i:i+1]))
            tensorlist.append(tf.scalar_mul(0.5, y_true[:, i:i + 1]))
        # digit_to_penalize = 0
        # digits = y_true[:,digit_to_penalize:digit_to_penalize+1]
        digits = tf.add_n(tensorlist)

        digit_activations = tf.multiply(digit_activations, digits)
        digit_activations = tf.scalar_mul(normalization, digit_activations)
        loss = tf.square(tf.subtract(digit_activations, norms))
        loss = tf.scalar_mul(alpha, loss)
        return loss

    SGDopt = keras.optimizers.SGD(lr=lr)
    rmsprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    with tf.device('/gpu:' + str(gpu)):
        model.compile(loss=[losses.binary_crossentropy, custom_loss], optimizer=rmsprop)
        # model.compile(loss=custom_loss, optimizer=rmsprop)
        # model.fit(x_train_0, [x_train_0, y_train_0_cat],
        model.fit(x_train, [x_train, y_train_cat],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, [x_test, y_test_cat]),
                  # validation_data=(x_test_0, [x_test_0, y_test_0_cat]),
                  callbacks=callbacks_list)


def Encoder(model):
    input_shape = (28, 28, 1)
    autoencoder = model
    encoder = Sequential()
    encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[1].get_weights(),
                       input_shape=input_shape))
    encoder.add(MaxPooling2D((2, 2), padding='same', weights=autoencoder.layers[2].get_weights()))
    encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[3].get_weights()))
    encoder.add(MaxPooling2D((2, 2), padding='same', weights=autoencoder.layers[4].get_weights()))
    encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[5].get_weights()))
    encoder.add(MaxPooling2D((2, 2), padding='same', weights=autoencoder.layers[6].get_weights()))
    encoded_imgs = encoder.predict(x_test)
    encoded_imgs = np.reshape(encoded_imgs, (-1, 128))
    #~ encoded_imgs.tofile('results/autoencoder_encoded_imgs.csv', sep=',', format='%10.5f')
    #~ encoded_from_file = genfromtxt('results/autoencoder_encoded_imgs.csv', delimiter=',')
    #~ encoded_from_file = np.reshape(encoded_from_file, (-1, 4, 4, 8))
    encoded_from_file = np.reshape(encoded_imgs, (-1, 4, 4, 8))
    return encoder,encoded_from_file


    # mean and covariance matrix of a certain digit


def get_digits(encoded_from_file,digit):
    bins = np.bincount(y_test)
    digits = np.zeros((bins[digit], 4, 4, 8))
    count = 0
    for i in range(np.shape(y_test)[0]):
        if y_test[i] == digit:
            digits[count] = encoded_from_file[i]
            count += 1
    return digits


def mean_cov(encoded_from_file, digit=0,lower_thresh=.2,upper_thresh=1):
    digits = get_digits(encoded_from_file,digit)
    digits = np.reshape(digits, (np.shape(digits)[0], 128))
    for i in range(10):
        print(LA.norm(digits[i][0:2], ord=2), 'norm')
    print(np.shape(digits), 'digits')
    good_digits = []
    for i in range(len(digits)):
        if upper_thresh > abs(LA.norm(digits[i][0:2], ord=2) - 5) > lower_thresh:
            good_digits.append(digits[i])
    good_digits = np.array(good_digits)
    print(np.shape(good_digits), 'good digits')
    # cov = np.cov(digits.T)
    # mean = np.mean(digits, axis=0)
    # samples = np.reshape(np.random.multivariate_normal(mean, cov, 100), (100, 4, 4, 8))

    cov = np.cov(good_digits.T)
    mean = np.mean(good_digits, axis=0)
    samples = np.reshape(np.random.multivariate_normal(mean, cov, 1000), (1000, 4, 4, 8))

    return mean,cov,samples


def Decoder(model, samples):
    autoencoder = model
    decoder = Sequential()
    decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[7].get_weights(),
                       input_shape=(4, 4, 8)))
    decoder.add(UpSampling2D((2, 2), weights=autoencoder.layers[8].get_weights()))
    decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same', weights=autoencoder.layers[9].get_weights()))
    decoder.add(UpSampling2D((2, 2), weights=autoencoder.layers[10].get_weights()))
    decoder.add(Conv2D(16, (3, 3), activation='relu', weights=autoencoder.layers[11].get_weights()))
    decoder.add(UpSampling2D((2, 2), weights=autoencoder.layers[12].get_weights()))
    decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', weights=autoencoder.layers[13].get_weights()))

    # decoded_from_file = decoder.predict(encoded_from_file)
    decoded_from_file = decoder.predict(samples)
    # decoded_mean = decoder.predict(np.reshape(mean, (1, 4, 4, 8)))

    return decoder,decoded_from_file


def save_images(decoded_imgs, epochs, digit, index, alpha):
    image_path = 'results/reg_autoencoder' + str(index) + '_epochs' + str(epochs) + '_digit' + str(
        digit) + '_alpha' + str(alpha) + '.png'
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(image_path)


if __name__ == '__main__' or True:
    epochs = 0
    digit = 0
    index = "testnew2"
    alpha = 1
    normalization = 10
    model, cnn = CreateModel()
    if epochs>0:
        train_model(model, epochs, index, normalization=normalization, gpu=1, batch_size=128, alpha=alpha, lr=0.002)
    encoder,encoded = Encoder(model)
    mean,cov,samples = mean_cov(encoded, digit)
    decoder,decoded = Decoder(model, samples)
    decoded2 = decoder.predict(samples)
    preds = cnn.predict(decoded)
    print(preds[0], 'preds')
    activation_mean = np.mean(preds, axis=0)[0]
    print(activation_mean, 'activation mean')
    activation_sd = np.std(preds, axis=0)[0]
    print(activation_sd, 'activation sd')
    #preds[0].tofile('results/predictions' + index + '.csv', sep=',', format='%10.5f')
    #activation_mean.tofile('results/activation_mean' + index + str(alpha) + '.csv', sep=',', format='%10.5f')
    #activation_sd.tofile('results/activation_sd' + index + str(alpha) + '.csv', sep=',', format='%10.5f')
    #save_images(decoded, epochs, digit, index, alpha)



