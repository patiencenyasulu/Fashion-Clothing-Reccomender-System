import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2

# Load the feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the sequence length (number of feature vectors per sequence)
seq_length = 10

# Create sequences of feature vectors
seq_list = []
for i in range(0, len(feature_list), seq_length):
    seq = feature_list[i:i+seq_length]
    if len(seq) == seq_length:
        seq_list.append(seq)

# Convert the sequences to numpy arrays
seq_array = np.array(seq_list)

# Define the target sequences (indices of the most similar images)
target_seqs = []
for i in range(len(seq_list)):
    query_vec = seq_list[i][-1]
    distances = norm(feature_list - query_vec, axis=1)
    indices = np.argsort(distances)[:6]
    target_seq = np.zeros(len(filenames))
    target_seq[indices] = 1
    target_seqs.append(target_seq)

# Convert the target sequences to numpy arrays
target_array = np.array(target_seqs)

# Define the BRT model
inputs = Input(shape=(seq_length, 2048))
x = LSTM(512, return_sequences=True)(inputs)
x = Dropout(0.5)(x)
x = LSTM(512)(x)
x = Dropout(0.5)(x)
x = Dense(len(filenames), activation='softmax')(x)
outputs = Reshape((len(filenames),))(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define the checkpoint callback to save the best model weights
checkpoint = ModelCheckpoint('brt_weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

# Train the model
model.fit(seq_array, target_array, batch_size=32, epochs=10, validation_split=0.2, callbacks=[checkpoint])

# Load the best model weights
model.load_weights('brt_weights.h5')

# Use the model to predict the target sequence for a new query image
img = image.load_img('sample/shirt.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
query_vec = model.predict(preprocessed_img)

# Find the most similar images based on the predicted target sequence
distances = norm(feature_list - query_vec, axis=1)
indices = np.argsort(distances)[:6]

# Display the most similar images
for file in filenames[indices]:
    temp_img = cv2.imread(file)
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)