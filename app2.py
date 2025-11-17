import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from config.paths import RAW_DATA_PATH


# -------------------------------
# Load Appliance Data from CSV
# -------------------------------
def load_appliance_data(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV Columns:", df.columns)
    print(df.head())
    return df

# -------------------------------
# Normalize and Denormalize
# -------------------------------
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    norm = 2 * (data - data_min) / (data_max - data_min) - 1
    return norm, data_min, data_max

def denormalize_data(norm_data, data_min, data_max):
    return (norm_data + 1) * (data_max - data_min) / 2 + data_min

# -------------------------------
# Generator with Conv1DTranspose
# -------------------------------
class Generator(tf.keras.Model):
    def __init__(self, segment_length):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1250, activation='relu', input_shape=(100,)),
            tf.keras.layers.Reshape((1250, 1)),
            
            tf.keras.layers.Conv1DTranspose(64, 25, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv1DTranspose(32, 25, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv1DTranspose(1, 25, strides=1, padding='same', activation='tanh'),
        ])
    
    def call(self, x):
        return self.model(x)

# -------------------------------
# Discriminator with Conv1D
# -------------------------------
class Discriminator(tf.keras.Model):
    def __init__(self, segment_length):
        super().__init__()
        self.model = tf.keras.Sequential([
            # Reshape to (segment_length, 1)
            tf.keras.layers.Reshape((segment_length, 1), input_shape=(segment_length,)),  # segment_length = 1000
            tf.keras.layers.Conv1D(64, 25, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(128, 25, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, x):
        return self.model(x)


# -------------------------------
# GAN Class
# -------------------------------
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    """def train_step(self, real_signals):
        batch_size = real_signals.shape[0]
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as d_tape:
            fake_signals = self.generator(noise, training=True)
            real_output = self.discriminator(real_signals, training=True)
            fake_output = self.discriminator(fake_signals, training=True)
            d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
            d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_signals = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_signals, training=True)
            g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True))

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
"""
    def train_step(self, real_signals):
          batch_size = real_signals.shape[0]
          noise = tf.random.normal([batch_size, 100])

          # Label smoothing
          real_labels = tf.ones_like(real_signals[:, :1]) * 0.9  # Real = 0.9
          fake_labels = tf.zeros_like(real_signals[:, :1]) + 0.1  # Fake = 0.1

          with tf.GradientTape() as d_tape:
              fake_signals = self.generator(noise, training=True)
              real_output = self.discriminator(real_signals, training=True)
              fake_output = self.discriminator(fake_signals, training=True)

              d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_output, from_logits=True)
              d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_output, from_logits=True)
              d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

          d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
          self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

          # Generator tries to fool the discriminator with smoothed labels (0.9)
          with tf.GradientTape() as g_tape:
              fake_signals = self.generator(noise, training=True)
              fake_output = self.discriminator(fake_signals, training=True)
              g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, fake_output, from_logits=True))

          g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
          self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

          return {"d_loss": d_loss, "g_loss": g_loss}

# -------------------------------
# Training Function
# -------------------------------
""" def train_gan(house_index, app_column, house_data, segment_length, epochs=100, batch_size=32):
    generator = Generator(segment_length)
    discriminator = Discriminator(segment_length)

    g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.05) # 0.0001, beta_1=0.05(fridge)
    d_optimizer = tf.keras.optimizers.Adam(0.00023, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)
    
    values = house_data.iloc[:, app_column].dropna().values
    MAX_POINTS = 50000  
    values = values[:min(len(values), MAX_POINTS)]
    values = values[:(len(values) // segment_length) * segment_length]  # Truncate to multiple of segment_length
    print(f"Training on {len(values)} points")

    segments = values.reshape(-1, segment_length)
    norm_segments = []
    min_max = []
    for seg in segments:
        norm, dmin, dmax = normalize_data(seg)
        norm_segments.append(norm)
        min_max.append((dmin, dmax))
    norm_segments = np.array(norm_segments)

    dataset = tf.data.Dataset.from_tensor_slices(norm_segments).shuffle(1024).batch(batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataset:
            losses = gan.train_step(batch)
        print(f"  D Loss: {losses['d_loss']:.4f} | G Loss: {losses['g_loss']:.4f}")

    # Generate one signal for evaluation
    noise = tf.random.normal((1, 100))
    gen_signal = generator(noise, training=False).numpy()[0]
    dmin, dmax = min_max[0]
    denorm_gen_signal = denormalize_data(gen_signal, dmin, dmax)
    real_signal = denormalize_data(norm_segments[0], dmin, dmax)

    plt.figure(figsize=(12, 6))
    plt.plot(real_signal, label="Real", color="blue")
    plt.plot(denorm_gen_signal, label="Generated", color="red", linestyle="--")
    plt.title(f"House {house_index} - Real vs Generated Signal")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(True)
    plt.show()"""
def train_gan(house_index, app_column, house_data, segment_length, epochs=100, batch_size=32):
    generator = Generator(segment_length)
    discriminator = Discriminator(segment_length)

    g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.05)  # 0.0001, beta_1=0.05 (fridge)
    d_optimizer = tf.keras.optimizers.Adam(0.00023, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)

    # ----------------------------
    # Prepare Data
    # ----------------------------
    values = house_data.iloc[:, app_column].dropna().values
    MAX_POINTS = 50000
    values = values[:min(len(values), MAX_POINTS)]
    values = values[:(len(values) // segment_length) * segment_length]  # Trim
    print(f"Training on {len(values)} points")

    segments = values.reshape(-1, segment_length)
    norm_segments = []
    min_max = []
    for seg in segments:
        norm, dmin, dmax = normalize_data(seg)
        norm_segments.append(norm)
        min_max.append((dmin, dmax))
    norm_segments = np.array(norm_segments)

    dataset = tf.data.Dataset.from_tensor_slices(norm_segments).shuffle(1024).batch(batch_size)

    # ----------------------------
    # Track losses
    # ----------------------------
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_d_loss = 0
        epoch_g_loss = 0
        steps = 0

        for batch in dataset:
            losses = gan.train_step(batch)
            epoch_d_loss += losses['d_loss']
            epoch_g_loss += losses['g_loss']
            steps += 1

        # Average over steps
        avg_d_loss = epoch_d_loss / steps
        avg_g_loss = epoch_g_loss / steps

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(f"  D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    # ----------------------------
    # Plot D Loss and G Loss
    # ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Generate and Compare Signal
    # ----------------------------
    noise = tf.random.normal((1, 100))
    gen_signal = generator(noise, training=False).numpy()[0]
    dmin, dmax = min_max[0]
    denorm_gen_signal = denormalize_data(gen_signal, dmin, dmax)
    real_signal = denormalize_data(norm_segments[0], dmin, dmax)

    plt.figure(figsize=(12, 6))
    plt.plot(real_signal, label="Real", color="blue")
    plt.plot(denorm_gen_signal, label="Generated", color="red", linestyle="--")
    plt.title(f"House {house_index} - Real vs Generated Signal")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 4  # Adjust as needed
    segment_length = 5000  # Set the segment length as per your needs

    csv_path = RAW_DATA_PATH / "HOUSE_1.csv"
    house_data = load_appliance_data(csv_path)

    if house_data is not None:
        train_gan(house_index, app_column, house_data, segment_length, epochs=200, batch_size=32)
