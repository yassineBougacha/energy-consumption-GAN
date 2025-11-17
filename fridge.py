"""import tensorflow as tf
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
            tf.keras.layers.Conv1D(128, 5, padding='same', input_shape=(100, 1)),
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
            tf.keras.layers.Conv1D(64, 5, padding='same', input_shape=(100, 1)),
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
# Save Generated Signal
# -------------------------------
def save_generated_signal(generator, house_index, filename="generated_signal_house"):
    noise = tf.random.normal(shape=(1, 100, 1))
    generated_signal = generator(noise, training=False).numpy()[0, :, 0]
    np.save(f"{filename}{house_index}.npy", generated_signal)
    print(f"Generated signal saved to {filename}{house_index}.npy")

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
    if len(values) < 100:
        print(f"Not enough data for House {house_index}, skipping...")
        return None

    real_apps = np.expand_dims(values[:100], axis=(0, -1))  # (1, 100, 1)

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
            if len(values) >= 100:
                all_values.append(values[:100])
            else:
                print(f"Skipping House {house_id}, not enough data.")
        except Exception as e:
            print(f"Error loading House {house_id}: {e}")

    if not all_values:
        print("No valid data found for global training.")
        return

    combined = np.stack(all_values, axis=0)  # shape (N, 100)
    global_data = np.mean(combined, axis=0)  # shape (100,)

    real_apps = np.expand_dims(global_data, axis=(0, -1))  # shape (1, 100, 1)

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
        save_generated_signal(generator, "global")

    print("Global GAN training complete.")

# -------------------------------
# Compare All Generated Signals
# -------------------------------
def compare_generated_signals(house_ids, include_global=True):
    plt.figure(figsize=(12, 6))
    for house_index in house_ids:
        try:
            generated_signal = np.load(f"generated_signal_house{house_index}.npy")
            plt.plot(generated_signal, label=f'House {house_index}')
        except FileNotFoundError:
            print(f"Generated signal for House {house_index} not found.")
    if include_global:
        try:
            global_signal = np.load("generated_signal_houseglobal.npy")
            plt.plot(global_signal, label="Global Field", linewidth=3, linestyle='--', color='black')
        except FileNotFoundError:
            print("Global generated signal not found.")
    plt.title("Comparison of Generated Signals")
    plt.xlabel("Time")
    plt.ylabel("Generated Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Main Training Script
# -------------------------------
if __name__ == "__main__":
    house_files = {
        1: {"file": RAW_DATA_PATH / "HOUSE_1.csv", "app_column": 1},
        2: {"file": RAW_DATA_PATH / "HOUSE_2.csv", "app_column": 1},
        5: {"file": RAW_DATA_PATH / "HOUSE_5.csv", "app_column": 1},
        6: {"file": RAW_DATA_PATH / "HOUSE_6.csv", "app_column": 1},
        9: {"file": RAW_DATA_PATH / "HOUSE_9.csv", "app_column": 1},
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
"""
"""import tensorflow as tf
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
        save_generated_signal(generator, "global")

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
            df = pd.read_csv("generated_signal_houseglobal.csv")
            global_signal = df["value"].values
            plt.figure(figsize=(10, 4))
            plt.plot(global_signal, label="Global Field", color='black')
            plt.title("Generated Signal - Global fridge Field")
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
        1: {"file": RAW_DATA_PATH / "HOUSE_1.csv", "app_column": 1},
        2: {"file": RAW_DATA_PATH / "HOUSE_2.csv", "app_column": 1},
        5: {"file": RAW_DATA_PATH / "HOUSE_5.csv", "app_column": 1},
        6: {"file": RAW_DATA_PATH / "HOUSE_6.csv", "app_column": 1},
        9: {"file": RAW_DATA_PATH / "HOUSE_9.csv", "app_column": 1},
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

"""
"""import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from config.paths import TRAINED_MODELS_PATH, RAW_DATA_PATH, SCALERS_PATH

# -------------------------------
# Load Appliance Data from CSV
# -------------------------------
def load_appliance_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("CSV Columns:", df.columns)
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return None

# -------------------------------
# Normalize the Data
# -------------------------------
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1  # Normalize to [-1, 1]

# -------------------------------
# Generator Model
# -------------------------------
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(5000,)),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(5000, activation='tanh')
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# Discriminator Model
# -------------------------------
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, input_shape=(5000,)),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# GAN Class
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
        with tf.GradientTape() as d_tape:
            fake_apps = self.generator(real_apps, training=True)
            real_output = self.discriminator(real_apps, training=True)
            fake_output = self.discriminator(fake_apps, training=True)
            d_loss = self.d_loss_fn(real_output, fake_output)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_apps = self.generator(real_apps, training=True)
            fake_output = self.discriminator(fake_apps, training=True)
            g_loss = self.g_loss_fn(fake_output)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# -------------------------------
# Save Generated Signal
# -------------------------------
def save_generated_signal(generator, house_index, filename="generated_signal_house"):
    noise = tf.random.normal(shape=(1, 5000))
    generated_signal = generator(noise, training=False).numpy()[0]
    df = pd.DataFrame(generated_signal, columns=["value"])
    df.to_csv(f"{filename}{house_index}.csv", index=False)
    print(f"Generated signal saved to {filename}{house_index}.csv")
    return generated_signal

# -------------------------------
# Train GAN Function
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

    values = house_data.iloc[:, app_column].dropna().values
    if len(values) < 5000:
        print(f"Not enough data for House {house_index}, skipping...")
        return None

    raw_signal = values[:5000]
    norm_signal = normalize_data(raw_signal)
    real_apps = np.expand_dims(norm_signal, axis=0)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} for House {house_index}")
        gan.train_step(real_apps)

    generated_signal = save_generated_signal(generator, house_index)
    return generated_signal, raw_signal  # Return raw real signal

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 3  # Adjust after printing the CSV

    house_data = load_appliance_data(RAW_DATA_PATH / "HOUSE_1.csv")

    if house_data is not None:
        print(f"\n=== Training GAN for House {house_index} ===")
        result = train_gan(house_index, app_column, house_data, epochs=10)

        if result:
            generated_signal, real_signal = result

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(real_signal, label="Real Signal (raw)", color="blue")
            plt.plot(generated_signal, label="Generated Signal", color="red", linestyle="--")
            plt.title(f"Comparison of Real and Generated Signals for House {house_index}")
            plt.xlabel("Time")
            plt.ylabel("Signal Value")
            plt.legend()
            plt.grid(True)

            # Print value ranges
            print("Real signal range:", np.min(real_signal), np.max(real_signal))
            print("Generated signal range:", np.min(generated_signal), np.max(generated_signal))

            plt.show()
"""
"""import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from config.paths import TRAINED_MODELS_PATH, RAW_DATA_PATH, SCALERS_PATH

# -------------------------------
# Load Appliance Data from CSV
# -------------------------------
def load_appliance_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("CSV Columns:", df.columns)
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return None

# -------------------------------
# Normalize the Data
# -------------------------------
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1  # Normalize to [-1, 1]

# -------------------------------
# Denormalize the Data
# -------------------------------
def denormalize_data(norm_data, data_min, data_max):
    return (norm_data + 1) * (data_max - data_min) / 2 + data_min

# -------------------------------
# Generator Model
# -------------------------------
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(100,)),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(5000, activation='tanh')
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# Discriminator Model
# -------------------------------
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, input_shape=(5000,)),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# -------------------------------
# GAN Class
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

    def train_step(self, noise, real_signal):
        with tf.GradientTape() as d_tape:
            fake_signal = self.generator(noise, training=True)
            real_output = self.discriminator(real_signal, training=True)
            fake_output = self.discriminator(fake_signal, training=True)
            d_loss = self.d_loss_fn(real_output, fake_output)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_signal = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_signal, training=True)
            g_loss = self.g_loss_fn(fake_output)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# -------------------------------
# Save Generated Signal
# -------------------------------
def save_generated_signal(generator, house_index, filename="generated_signal_house"):
    noise = tf.random.normal(shape=(1, 100))
    generated_signal = generator(noise, training=False).numpy()[0]
    df = pd.DataFrame(generated_signal, columns=["value"])
    df.to_csv(f"{filename}{house_index}.csv", index=False)
    print(f"Generated signal saved to {filename}{house_index}.csv")
    return generated_signal

# -------------------------------
# Train GAN Function
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

    values = house_data.iloc[:, app_column].dropna().values
    if len(values) < 5000:
        print(f"Not enough data for House {house_index}, skipping...")
        return None

    raw_signal = values[:5000]
    norm_signal = normalize_data(raw_signal)
    real_signal = np.expand_dims(norm_signal, axis=0)

    for epoch in range(epochs):
        noise = tf.random.normal(shape=(1, 100))
        print(f"Epoch {epoch + 1}/{epochs} for House {house_index}")
        gan.train_step(noise, real_signal)

    generated_signal = save_generated_signal(generator, house_index)
    return generated_signal, raw_signal  # Return raw real signal

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 3  # Adjust after printing the CSV

    house_data = load_appliance_data(RAW_DATA_PATH / "HOUSE_1.csv")

    if house_data is not None:
        print(f"\n=== Training GAN for House {house_index} ===")
        result = train_gan(house_index, app_column, house_data, epochs=10)

        if result:
            generated_signal, real_signal = result

            # Dénormaliser les signaux générés
            data_min = np.min(real_signal)
            data_max = np.max(real_signal)
            denormalized_generated_signal = denormalize_data(generated_signal, data_min, data_max)

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(real_signal, label="Real Signal (raw)", color="blue")
            plt.plot(denormalized_generated_signal, label="Generated Signal (denormalized)", color="red", linestyle="--")
            plt.title(f"Comparison of Real and Generated Signals for House {house_index}")
            plt.xlabel("Time")
            plt.ylabel("Signal Value")
            plt.legend()
            plt.grid(True)

            # Print value ranges
            print("Real signal range:", np.min(real_signal), np.max(real_signal))
            print("Generated signal range:", np.min(denormalized_generated_signal), np.max(denormalized_generated_signal))

            plt.show()"""


"""import tensorflow as tf
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
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1250, activation='relu', input_shape=(100,)),
            tf.keras.layers.Reshape((1250, 1)),
            tf.keras.layers.Conv1DTranspose(64, 25, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv1DTranspose(32, 25, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv1DTranspose(1, 25, strides=1, padding='same', activation='tanh'),
            tf.keras.layers.Reshape((5000,))
        ])
    def call(self, x):
        return self.model(x)

# -------------------------------
# Discriminator with Conv1D
# -------------------------------
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape((5000, 1), input_shape=(5000,)),
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

    def train_step(self, real_signals):
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

# -------------------------------
# Training Function
# -------------------------------
def train_gan(house_index, app_column, house_data, epochs=100, batch_size=32):
    generator = Generator()
    discriminator = Discriminator()

    g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)

    values = house_data.iloc[:, app_column].dropna().values
    values = values[:(len(values) // 5000) * 5000]  # Truncate to multiple of 5000
    print(f"Training on {len(values)} points")

    segments = values.reshape(-1, 5000)
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
    plt.show()

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 3  # Adjust as needed

    csv_path = RAW_DATA_PATH / "HOUSE_1.csv"
    house_data = load_appliance_data(csv_path)

    if house_data is not None:
        train_gan(house_index, app_column, house_data, epochs=100, batch_size=32)

"""
"""import tensorflow as tf
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

    def train_step(self, real_signals):
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
def train_gan(house_index, app_column, house_data, segment_length, epochs=100, batch_size=32):
    generator = Generator(segment_length)
    discriminator = Discriminator(segment_length)

    g_optimizer = tf.keras.optimizers.Adam(0.00005, beta_1=0.05) # 0.0001, beta_1=0.05(fridge)
    d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)
    
    values = house_data.iloc[:, app_column].dropna().values
    MAX_POINTS = 10000  
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
    plt.show()

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 3  # Adjust as needed
    segment_length = 5000  # Set the segment length as per your needs

    csv_path = RAW_DATA_PATH / "HOUSE_1.csv"
    house_data = load_appliance_data(csv_path)

    if house_data is not None:
        train_gan(house_index, app_column, house_data, segment_length, epochs=650, batch_size=32)

"""
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
            tf.keras.layers.Reshape((segment_length, 1), input_shape=(segment_length,)),
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

    def train_step(self, real_signals):
        batch_size = real_signals.shape[0]
        noise = tf.random.normal([batch_size, 100])

        # Label smoothing
        real_labels = tf.ones_like(real_signals[:, :1]) * 0.9
        fake_labels = tf.zeros_like(real_signals[:, :1]) + 0.1

        with tf.GradientTape() as d_tape:
            fake_signals = self.generator(noise, training=True)
            real_output = self.discriminator(real_signals, training=True)
            fake_output = self.discriminator(fake_signals, training=True)

            d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_output, from_logits=True)
            d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_output, from_logits=True)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

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
"""def train_gan(house_index, app_column, house_data, segment_length, epochs=100, batch_size=32):
    generator = Generator(segment_length)
    discriminator = Discriminator(segment_length)

    g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.05)
    d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)
    
    values = house_data.iloc[:, app_column].dropna().values
    MAX_POINTS = 30000
    values = values[:min(len(values), MAX_POINTS)]
    values = values[:(len(values) // segment_length) * segment_length]
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

    # Output directory for signals
    output_dir = Path("generated_signals")
    output_dir.mkdir(exist_ok=True)

    # Store all generated signals
    all_generated_signals = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataset:
            losses = gan.train_step(batch)
        print(f"  D Loss: {losses['d_loss']:.4f} | G Loss: {losses['g_loss']:.4f}")

        # Save one signal each epoch
        noise = tf.random.normal((1, 100))
        gen_signal = generator(noise, training=False).numpy()[0].squeeze()
        dmin, dmax = min_max[0]
        denorm_gen_signal = denormalize_data(gen_signal, dmin, dmax)
        all_generated_signals.append(denorm_gen_signal)

        # Optionally plot a preview
        if epoch == epochs - 1:  # Just plot the last one
            real_signal = denormalize_data(norm_segments[0], dmin, dmax)
            plt.figure(figsize=(12, 6))
            plt.plot(real_signal, label="Real", color="blue")
            plt.plot(denorm_gen_signal, label="Generated", color="red", linestyle="--")
            plt.title(f"House {house_index} - Real vs Generated Signal (Epoch {epoch + 1})")
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Save all generated signals
    all_generated_signals = np.array(all_generated_signals)
    np.save(output_dir / f"house{house_index}_all_gen_signals.npy", all_generated_signals)

    # Save as CSV
    df_all = pd.DataFrame(all_generated_signals.T)
    df_all.to_csv(output_dir / f"house{house_index}_all_gen_signalss.csv", index=False)
    print(f"Saved all generated signals to {output_dir}")"""




    # -------------------------------
# Training Function
# -------------------------------
def train_gan(house_index, app_column, house_data, segment_length, epochs=100, batch_size=32):
    generator = Generator(segment_length)
    discriminator = Discriminator(segment_length)

    g_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.05)
    d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    gan = GAN(generator, discriminator, g_optimizer, d_optimizer)
    
    values = house_data.iloc[:, app_column].dropna().values
    MAX_POINTS = 30000
    values = values[:min(len(values), MAX_POINTS)]
    values = values[:(len(values) // segment_length) * segment_length]
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

    # Output directory for signals
    output_dir = Path("generated_signals")
    output_dir.mkdir(exist_ok=True)

    all_generated_signals = []

    # Track losses
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataset:
            losses = gan.train_step(batch)
        print(f"  D Loss: {losses['d_loss']:.4f} | G Loss: {losses['g_loss']:.4f}")
        d_losses.append(losses['d_loss'].numpy())
        g_losses.append(losses['g_loss'].numpy())

        noise = tf.random.normal((1, 100))
        gen_signal = generator(noise, training=False).numpy()[0].squeeze()
        dmin, dmax = min_max[0]
        denorm_gen_signal = denormalize_data(gen_signal, dmin, dmax)
        all_generated_signals.append(denorm_gen_signal)

        if epoch == epochs - 1:
            real_signal = denormalize_data(norm_segments[0], dmin, dmax)
            plt.figure(figsize=(12, 6))
            plt.plot(real_signal, label="Real", color="blue")
            plt.plot(denorm_gen_signal, label="Generated", color="red", linestyle="--")
            plt.title(f"House {house_index} - Real vs Generated Signal (Epoch {epoch + 1})")
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Save all generated signals
    all_generated_signals = np.array(all_generated_signals)
    np.save(output_dir / f"house{house_index}_all_gen_signals.npy", all_generated_signals)

    df_all = pd.DataFrame(all_generated_signals.T)
    df_all.to_csv(output_dir / f"house{house_index}_all_gen_signalss.csv", index=False)
    print(f"Saved all generated signals to {output_dir}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    house_index = 1
    app_column = 3  # Adjust as needed
    segment_length = 5000  # Set your segment length

    csv_path = RAW_DATA_PATH / "HOUSE_1.csv"
    house_data = load_appliance_data(csv_path)

    if house_data is not None:
        train_gan(house_index, app_column, house_data, segment_length, epochs=250, batch_size=32)

