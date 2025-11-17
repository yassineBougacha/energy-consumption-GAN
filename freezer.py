import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config.paths import TRAINED_MODELS_PATH, RAW_DATA_PATH, SCALERS_PATH

# -------------------------------
# Load Appliance Data from CSV
# -------------------------------
def load_appliance_data(csv_path):
    return pd.read_csv(csv_path)

# -------------------------------
# Generator Model Definition
# -------------------------------
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 5, padding='same', input_shape=(1000, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv1D(256, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv1D(512, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv1D(1, 3, padding='same', activation='relu')
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# Discriminator Model Definition
# -------------------------------
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 5, padding='same', input_shape=(1000, 1)),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(128, 5, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# GAN Model
# -------------------------------
class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, d_loss_fn, g_loss_fn):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, real_apps):
        # Train Discriminator
        with tf.GradientTape() as d_tape:
            fake_apps = self.generator(real_apps, training=True)
            real_output = self.discriminator(real_apps, training=True)
            fake_output = self.discriminator(fake_apps, training=True)
            d_loss = self.d_loss_fn(real_output, fake_output)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as g_tape:
            fake_apps = self.generator(real_apps, training=True)
            fake_output = self.discriminator(fake_apps, training=True)
            g_loss = self.g_loss_fn(fake_output)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# -------------------------------
# Save Generated Signal as CSV
# -------------------------------
def save_generated_signal(generator, house_index, filename="generated_signal_house"):
    noise = tf.random.normal(shape=(1, 1000, 1))
    generated_signal = generator(noise, training=False).numpy()[0, :, 0]
    df = pd.DataFrame(generated_signal, columns=["value"])
    df.to_csv(f"{filename}{house_index}.csv", index=False)
    print(f"Generated signal saved to {filename}{house_index}.csv")

# -------------------------------
# Train GAN for a Single Appliance
# -------------------------------
def train_gan(house_index, app_column, house_data, epochs=10):
    generator = Generator()
    discriminator = Discriminator()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def d_loss_fn(real, fake):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake))) + \
               tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.zeros_like(real)))

    def g_loss_fn(fake):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer, d_loss_fn, g_loss_fn)

    values = house_data.iloc[:, app_column].values
    if len(values) < 1000:
        print(f"Not enough data for House {house_index}, skipping...")
        return None

    real_apps = np.expand_dims(values[:1000], axis=(0, -1))  # (1, 1000, 1)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} for House {house_index}")
        gan.train_step(real_apps)
        save_generated_signal(generator, house_index)

    return generator

# -------------------------------
# Train Global GAN on Combined Data
# -------------------------------
def train_global_gan(house_files, epochs=10):
    print("\n=== Training Global GAN on Combined Appliance Data ===")
    all_values = []

    for house_id, info in house_files.items():
        try:
            data = load_appliance_data(info["file"])
            values = data.iloc[:, info["app_column"]].dropna().values
            if len(values) >= 1000:
                all_values.append(values[:1000])
            else:
                print(f"Skipping House {house_id}, not enough data.")
        except Exception as e:
            print(f"Error loading House {house_id}: {e}")

    if not all_values:
        print("No valid data found for global training.")
        return

    combined = np.stack(all_values, axis=0)  # shape (N, 1000)
    global_data = np.mean(combined, axis=0)  # shape (1000,)

    real_apps = np.expand_dims(global_data, axis=(0, -1))  # shape (1, 1000, 1)

    generator = Generator()
    discriminator = Discriminator()

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def d_loss_fn(real, fake):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake))) + \
               tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.zeros_like(real)))

    def g_loss_fn(fake):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer, d_loss_fn, g_loss_fn)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} for Global Field")
        gan.train_step(real_apps)
        save_generated_signal(generator, "globalfreezer")

    print("Global GAN training complete.")

# -------------------------------
# Compare All Generated Signals (separate plots)
# -------------------------------
def compare_generated_signals(house_ids, include_global=True):
    for house_index in house_ids:
        try:
            df = pd.read_csv(f"generated_signal_house{house_index}.csv")
            generated_signal = df["value"].values
            plt.figure(figsize=(10, 4))
            plt.plot(generated_signal, label=f'House {house_index}')
            plt.title(f"Generated Signal - House {house_index}")
            plt.xlabel("Time")
            plt.ylabel("Generated Value")
            plt.legend()
            plt.grid(True)
        except FileNotFoundError:
            print(f"Generated signal for House {house_index} not found.")

    if include_global:
        try:
            df = pd.read_csv("generated_signal_houseglobalfreezer.csv")
            global_signal = df["value"].values
            plt.figure(figsize=(10, 4))
            plt.plot(global_signal, label="Global Field", color='black')
            plt.title("Generated Signal - Global freezer Field")
            plt.xlabel("Time")
            plt.ylabel("Generated Value")
            plt.legend()
            plt.grid(True)
        except FileNotFoundError:
            print("Global generated signal not found.")

    plt.show()  # Show all figures together

# -------------------------------
# Main Training Script
# -------------------------------
if __name__ == "__main__":
    house_files = {
        #1: {"file": RAW_DATA_PATH / "HOUSE_1.csv", "app_column": 5},
        2: {"file": RAW_DATA_PATH / "HOUSE_2.csv", "app_column": 2},
        3: {"file": RAW_DATA_PATH / "HOUSE_3.csv", "app_column": 1},
        4: {"file": RAW_DATA_PATH / "HOUSE_4.csv", "app_column": 1},
        5: {"file": RAW_DATA_PATH / "HOUSE_5.csv", "app_column": 2},
        6: {"file": RAW_DATA_PATH / "HOUSE_6.csv", "app_column": 2},
    }

    trained_house_ids = []

    for house_id, info in house_files.items():
        print(f"\n=== Training GAN for House {house_id} ===")
        data = load_appliance_data(info["file"])
        gen = train_gan(house_id, info["app_column"], data, epochs=10)
        if gen is not None:
            trained_house_ids.append(house_id)

    print("\n=== Training Completed for Individual Houses ===")
    train_global_gan(house_files, epochs=10)

    print("\n=== Plotting Comparison ===")
    compare_generated_signals(trained_house_ids, include_global=True)

