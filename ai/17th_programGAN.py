import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Discriminator model
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Combine generator and discriminator into a GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile models
def compile_models(generator, discriminator, gan, latent_dim):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

# Generate random noise for the generator
def generate_latent_points(latent_dim, n_samples):
    return np.random.normal(0, 1, (n_samples, latent_dim))

# Generate fake samples using the generator
def generate_fake_samples(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    fake_samples = generator.predict(latent_points)
    return fake_samples

# Train the GAN
def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=100, batch_size=128):
    batch_per_epoch = dataset.shape[0] // batch_size
    half_batch = batch_size // 2

    for epoch in range(n_epochs):
        for batch in range(batch_per_epoch):
            # Train discriminator on real samples
            idx = np.random.randint(0, dataset.shape[0], half_batch)
            real_samples = dataset[idx]
            labels_real = np.ones((half_batch, 1))

            d_loss_real = discriminator.train_on_batch(real_samples, labels_real)

            # Train discriminator on fake samples
            fake_samples = generate_fake_samples(generator, latent_dim, half_batch)
            labels_fake = np.zeros((half_batch, 1))

            d_loss_fake = discriminator.train_on_batch(fake_samples, labels_fake)

            # Train generator
            latent_points = generate_latent_points(latent_dim, batch_size)
            labels_gan = np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(latent_points, labels_gan)

            print(f"{epoch + 1}/{n_epochs}, {batch}/{batch_per_epoch}, D_real={d_loss_real[0]}, D_fake={d_loss_fake[0]}, G={g_loss}")

# Example usage
latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

compile_models(generator, discriminator, gan, latent_dim)

# Load and preprocess your dataset (e.g., MNIST)
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_train = np.expand_dims(X_train, axis=-1)

# Train the GAN
train_gan(generator, discriminator, gan, X_train, latent_dim)
