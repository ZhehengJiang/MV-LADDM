
import tensorflow as tf

from keras.layers import Lambda, Dense, Concatenate, Reshape, Layer
import keras

class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


class MLVAE(Layer):
    def __init__(self, feature_dim, latent_dim, num_channels, sequence_length, **kwargs):
        super(MLVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.z_mean_layers = []
        self.z_log_var_layers = []
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.decoder = []
        for i in range(num_channels):
            self.z_mean_layers.append(Dense(self.latent_dim, name="z_mean_{}".format(str(i))))
            self.z_log_var_layers.append(Dense(self.latent_dim, name="z_log_var_{}".format(str(i))))
            self.decoder.append(Dense(self.feature_dim, name="decoder_{}".format(str(i))))

    def elbo_loss(self, recon_x1, x1, recon_x2, x2, mu, logvar,
                  lambda_image=1.0, lambda_text=1.0, annealing_factor=1):

        x1_bce, x2_bce = 0, 0
        if recon_x1 is not None and x1 is not None:
            x1_bce = keras.backend.sum(keras.losses.mean_squared_error(
                recon_x1, x1))

        if recon_x2 is not None and x2 is not None:
            x2_bce = keras.backend.sum(keras.losses.mean_squared_error(recon_x2, x2))

        KLD = -0.5 * keras.backend.sum(1 + logvar - keras.backend.square(mu) - keras.backend.exp(logvar))
        ELBO = keras.backend.mean(lambda_image * x1_bce + lambda_text * x2_bce
                          + annealing_factor * KLD)
        return ELBO

    def call(self, branches):
        z_sequence = []
        loss = 0

        for i in range(self.sequence_length):
            z_mean_list = []
            z_log_var_list = []
            inputs = []
            for j in range(self.num_channels):
                out = Lambda(lambda x: x[:, i, :])(branches[j])
                inputs.append(out)
                z_mean_list.append(Reshape((1,self.latent_dim))(self.z_mean_layers[j](out)))
                z_log_var_list.append(Reshape((1,self.latent_dim))(self.z_log_var_layers[j](out)))
            z_means = Concatenate(axis=1)(z_mean_list)
            z_log_vars = Concatenate(axis=1)(z_log_var_list)
            eps = keras.backend.epsilon()
            var = keras.backend.exp(z_log_vars) + eps
            T = 1. / (var + eps)
            mu = keras.backend.sum(z_means * T, axis=1) / keras.backend.sum(T, axis=1)
            pd_var = 1. / keras.backend.sum(T, axis=1)
            logvar = keras.backend.log(pd_var + eps)
            z = Sampling()([mu, logvar])
            recons = []
            for jj in range(self.num_channels):
                recons.append(self.decoder[jj](z))
            loss = loss + self.elbo_loss(recons[0],inputs[0],recons[1],inputs[1],mu,logvar)
            z_sequence.append(Reshape((1,self.latent_dim))(z))
        loss = loss/self.sequence_length
        z_sequence = Concatenate(axis=1)(z_sequence)
        return [z_sequence, loss]


