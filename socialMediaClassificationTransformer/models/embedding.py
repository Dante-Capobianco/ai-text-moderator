#Code for word/positional vector embeddings of tokenized text input
#Create matrix of vocab size x embedding dimension, pass in array of tokenized text, output matrix of array size x embedding size
#Trainable
#Output is Bxmax lengthxembedding size
import numpy as np


from socialMediaClassificationTransformer.training.config import DIMENSION, MAX_TOKEN_SIZE, BATCH_SIZE, VOCAB_SIZE, \
    LEARNING_RATE, DROPOUT_RATE

import pickle
import tensorflow as tf
import random

def load_and_prepare_data(file_path, batch_size):
    # Load the data from the pickle file
    data = load_data(file_path)

    # Randomize the order of the data
    random.shuffle(data)

    # Grab only the first 'batch_size' entries
    batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    return batch_data

def load_data(file_path):
    with open(file_path, 'rb') as file:
        try:
            return pickle.load(file)
        except EOFError as e:
            raise EOFError(f"Error loading {file_path}: {e}")


def get_positional_encoding(seq_len, d_model):
    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-np.log(10000.0) / d_model))

    # Apply sine to even indices
    sin_term = tf.sin(position * div_term)

    # Apply cosine to odd indices
    cos_term = tf.cos(position * div_term)

    # Interleave sin and cos terms
    pe = tf.concat([sin_term, cos_term], axis=-1)

    # Ensure that the pe tensor has the right shape if d_model is odd
    pe = tf.reshape(pe, (seq_len, d_model))

    return pe

def embedding_lookup(token_ids, embedding_weights):
    return embedding_weights[token_ids]

def forward_pass(batch_data, embedding_weights, training=True):
    # Extract token IDs from batch data
    token_ids = np.array([entry['token_ids'] for entry in batch_data])

    # Lookup embeddings
    embeddings = embedding_lookup(token_ids, embedding_weights)
    sum_embeddings = embeddings + get_positional_encoding(MAX_TOKEN_SIZE, DIMENSION)

    if training:
        sum_embeddings = dropout(sum_embeddings, DROPOUT_RATE)

    return sum_embeddings

def backward_pass(embedding_weights, gradient):

    # Update the embedding weights
    embedding_weights -= LEARNING_RATE * gradient
    return embedding_weights

def init_embedding_weights():
    return np.random.uniform(-0.5, 0.5, (VOCAB_SIZE, DIMENSION))

def init_embedding(file_path):
    try:
        return load_and_prepare_data(file_path, BATCH_SIZE)

    except Exception as e:
        print(f"Failed to process: {e}")

def dropout(x, rate):
    retain_prob = 1.0 - rate
    mask = np.random.binomial(1, retain_prob, size=x.shape)
    return x * mask / retain_prob