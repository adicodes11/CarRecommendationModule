import numpy as np
import tensorflow as tf
from keras import layers, Model, Input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
car_data = pd.read_csv(r'C:\Users\Aditya\Desktop\Arundhati\car_recommendation_project\dataset.csv')

# Clean column names
car_data.columns = car_data.columns.str.strip()

# Handle missing values in the dataset (replace NaNs with 'Unknown' for categorical and 0 for numerical)
car_data.fillna({'Fuel Type': 'Unknown', 'Body Type': 'Unknown', 'Transmission Type': 'Unknown'}, inplace=True)
car_data.fillna(0, inplace=True)

# Select features for VAE
selected_features = ['Ex-Showroom Price', 'Fuel Type', 'Body Type', 'Transmission Type']

# Separate numeric and categorical features
numeric_features = ['Ex-Showroom Price']
categorical_features = ['Fuel Type', 'Body Type', 'Transmission Type']

# Preprocessing for numeric data (scaling)
numeric_transformer = MinMaxScaler()

# Preprocessing for categorical data (one-hot encoding)
categorical_transformer = OneHotEncoder(sparse_output=False)

# Combine preprocessing steps for both types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to the selected features
x_train = preprocessor.fit_transform(car_data[selected_features])

# Set input dimensions according to the selected features
input_dim = x_train.shape[1]
latent_dim = 2  # Latent space dimension

# Define and register the sampling function
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder model
encoder_input = Input(shape=(input_dim,), name="encoder_input")
x = layers.Dense(512, activation='relu')(encoder_input)
x = layers.BatchNormalization()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

# Decoder model
decoder_input = Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(512, activation='relu')(decoder_input)
x = layers.BatchNormalization()(x)
decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)
decoder = Model(decoder_input, decoder_output, name='decoder')

# VAE model
vae_output = decoder(encoder(encoder_input)[2])  # Taking 'z' from encoder output
vae = Model(encoder_input, vae_output, name='vae')

# Custom VAE loss function
def vae_loss(x, x_decoded_mean, z_log_var, z_mean):
    reconstruction_loss = tf.keras.backend.mean(tf.square(x - x_decoded_mean), axis=-1)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
    )
    return reconstruction_loss + kl_loss

# Custom loss function to pass to the model
def custom_vae_loss(x, x_decoded_mean):
    z_mean_output, z_log_var_output = encoder(x)[:2]
    return vae_loss(x, x_decoded_mean, z_log_var_output, z_mean_output)

# Compile the VAE model
vae.compile(optimizer='adam', loss=custom_vae_loss)

# Train the VAE model
vae.fit(x_train, x_train, epochs=50, batch_size=64, validation_split=0.1)

# Save the models
vae.save('my_model.keras')
encoder.save('vae_encoder.h5')
decoder.save('vae_decoder.h5')

print("Models saved successfully.")
