#Implementation of FNN with 2 linear transformations & an activation function in-between them, possibly with gated linear unit

#Add dropout to output before adding residual connection
#He initialization
import tensorflow as tf

from socialMediaClassificationTransformer.training.config import DROPOUT_RATE, DIMENSION, NUM_LAYERS


#Set up ffn weights for each hidden layer in the network, using He initialization due to using an activation function
def init_ffn_weights():
    hidden_dim = 4 * DIMENSION
    ffn_weights = []
    for _ in range(NUM_LAYERS):
        W_1 = tf.Variable(initial_value=tf.random.truncated_normal((DIMENSION, 2 * hidden_dim), stddev=tf.sqrt(2.0 / tf.cast(DIMENSION, tf.float32))),
                          trainable=True  # Indicates that the variable should be trainable

                          )
        W_2 = tf.Variable(initial_value=tf.random.truncated_normal((hidden_dim, DIMENSION), stddev=tf.sqrt(2.0 / tf.cast(hidden_dim, tf.float32))),
        trainable=True)  # Indicates that the variable should be trainable
        ffn_weights.append((W_1, W_2))
    return ffn_weights

def geglu(x):
    """Gated Linear Unit with GELU. Uses GELU activation due to non-zero values returned for negative inputs
    & a smooth, non-linear curve. The gating ensures computational resources are focused on more informative
    tokens in each sequence, using the GELU activation on part of the embedding dimension to identify if the
    token is of greater significant (closer to 1) or not (closer to 0). It enhances & diminishes the dimensions of
    tokens based on their relevance"""
    x, gate = tf.split(x, num_or_size_splits=2, axis=-1)
    return x * tf.nn.gelu(gate)

#Apply 2 linear transformations with a GeGLU & dropout operation in between
@tf.function
def feedforward_network(input, w1, w2, training=True):

    # First Linear Transformation
    hidden_output = tf.matmul(input, w1)  # Shape: (batch_size, token_size, hidden_dim)

    # Apply GeGLU Activation - half through gating mechanism, other not through (split embedding dimension in half)
    activation_output = geglu(hidden_output)  # Shape: (batch_size, token_size, hidden_dim)

    #Apply during training to minimize over reliance
    activation_output = tf.nn.dropout(activation_output, rate=DROPOUT_RATE) if training else activation_output


    # Second Linear Transformation
    output = tf.matmul(activation_output, w2)  # Shape: (batch_size, token_size, embedding_size)

    return output