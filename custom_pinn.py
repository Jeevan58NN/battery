import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.d1 = tf.keras.layers.Dense(units, activation='tanh')
        self.d2 = tf.keras.layers.Dense(units, activation='tanh')
        self.skip = tf.keras.layers.Dense(units)

    def call(self, x):
        h = self.d1(x)
        h = self.d2(h)
        return h + self.skip(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@tf.keras.utils.register_keras_serializable()
class PINN(tf.keras.Model):
    def __init__(self, input_dim=15, output_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.inp = tf.keras.layers.InputLayer(input_shape=(input_dim,))
        self.b1 = ResidualBlock(128)
        self.b2 = ResidualBlock(128)
        self.b3 = ResidualBlock(128)
        self.out = tf.keras.layers.Dense(output_dim)

        self.k = tf.constant(1.0, dtype=tf.float32)

    def call(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        y = self.out(x)

        V = y[:, 0:1]
        I = y[:, 1:2]
        E = y[:, 2:3]
        S = y[:, 3:4]

        energy_constraint = E - self.k * V * I
        self.add_loss(tf.reduce_mean(tf.square(energy_constraint)))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        })
        return config
