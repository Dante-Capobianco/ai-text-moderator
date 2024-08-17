#Implementation of self-attention & multi-head attention mechanisms

from socialMediaClassificationTransformer.training.config import ATTENTION_HEADS, DIMENSION, LEARNING_RATE, DROPOUT_RATE
import numpy as np

def xavier_initialization(shape, uniform=True):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))

    if uniform:
        return np.random.uniform(-limit, limit, size=shape)
    else:
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)

def init_attention_weights():
    assert DIMENSION % ATTENTION_HEADS == 0
    #Consider He initialization; currently using xavier
    return {
        "W_q": xavier_initialization((DIMENSION, DIMENSION)),
        "W_k": xavier_initialization((DIMENSION, DIMENSION)),
        "W_v": xavier_initialization((DIMENSION, DIMENSION)),
        "W_o": xavier_initialization((DIMENSION, DIMENSION)),
    }

def update_weights(weights, gradients):
    for key in weights:
        weights[key] -= LEARNING_RATE * gradients[key]
    return weights

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def multi_head_self_attention(inputs, attention_weights, training=True):
    head_dim = DIMENSION // ATTENTION_HEADS
    num_heads = ATTENTION_HEADS


    # Linear projections
    Q = np.dot(inputs, attention_weights["W_q"])  # Shape: (batch_size, sequence_length, embedding_dim)
    K = np.dot(inputs, attention_weights["W_k"])  # Shape: (batch_size, sequence_length, embedding_dim)
    V = np.dot(inputs, attention_weights["W_v"])  # Shape: (batch_size, sequence_length, embedding_dim)# Split the embedding dimension into multiple heads

    # Possible experimentation: Apply GELU activation function
    # Q = gelu(Q)
    # K = gelu(K)
    # V = gelu(V)

    Q = np.reshape(Q, (-1, Q.shape[1], num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)
    K = np.reshape(K, (-1, K.shape[1], num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)
    V = np.reshape(V, (-1, V.shape[1], num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)# Compute the scaled dot-product attention for each head

    attention_scores = np.einsum('bqhd,bkhd->bhqk', Q, K)  # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    attention_scores = attention_scores / np.sqrt(head_dim)
    # Apply the stability fix by subtracting the max value
    max_attention_scores = np.max(attention_scores, axis=-1, keepdims=True)
    attention_scores -= max_attention_scores

    #Apply softmax to get attention weights
    score_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)

    attention_output = np.einsum('bhqk,bkhd->bqhd', score_weights, V)  # Shape: (batch_size, sequence_length, num_heads, head_dim)# Concatenate the heads

    if training:
        # Apply dropout to attention output (before residual connection)
        attention_output = dropout(attention_output, DROPOUT_RATE)

    attention_output = np.reshape(attention_output, (-1, attention_output.shape[1], DIMENSION))  # Shape: (batch_size, sequence_length, embedding_dim)# Final linear layer


    # Shape: (batch_size, sequence_length, embedding_dim)
    output = np.dot(attention_output, attention_weights["W_o"])
    return output

def dropout(x, rate):
    retain_prob = 1.0 - rate
    mask = np.random.binomial(1, retain_prob, size=x.shape)
    return x * mask / retain_prob