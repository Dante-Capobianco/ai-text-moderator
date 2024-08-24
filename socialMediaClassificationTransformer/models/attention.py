#Implementation of self-attention & multi-head attention mechanisms
from tensorflow.python.keras.utils.layer_utils import print_summary

from socialMediaClassificationTransformer.training.config import ATTENTION_HEADS, DIMENSION, LEARNING_RATE, \
    DROPOUT_RATE, MAX_TOKEN_SIZE, NUM_LAYERS
import tensorflow as tf

#Used for operations without an activation function
@tf.function
def xavier_initialization(shape, uniform=True):
    fan_in, fan_out = shape[0], shape[1]
    limit = tf.sqrt(6.0 / (tf.cast(fan_in, tf.float32) + tf.cast(fan_out, tf.float32)))

    if uniform:
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32)
    else:
        stddev = tf.sqrt(2.0 / (tf.cast(fan_in, tf.float32) + tf.cast(fan_out, tf.float32)))
        return tf.random.normal(shape, mean=0.0, stddev=stddev, dtype=tf.float32)

#Initialize attention weights for query, key, value, and output operations; 4 unique weights for each layer
def init_attention_weights():
    assert DIMENSION % ATTENTION_HEADS == 0
    attention_weights = []
    #Consider He initialization; currently using xavier
    for _ in range(NUM_LAYERS):
        layer_weights = {
            "W_q": tf.Variable(initial_value=xavier_initialization((DIMENSION, DIMENSION)),trainable=True),
            "W_k": tf.Variable(initial_value=xavier_initialization((DIMENSION, DIMENSION)),trainable=True),
            "W_v": tf.Variable(initial_value=xavier_initialization((DIMENSION, DIMENSION)),trainable=True),
            "W_o": tf.Variable(initial_value=xavier_initialization((DIMENSION, DIMENSION)),trainable=True),
        }
        attention_weights.append(layer_weights)
    return attention_weights

@tf.function
def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function (if proceeding with activation within attention)
    """
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / tf.constant(3.141592653589793)) * (x + 0.044715 * tf.pow(x, 3))))

#Multi-head, masked self-attention mechanism
@tf.function
def multi_head_self_attention(inputs, attention_weights, attention_mask, training=True):
    head_dim = DIMENSION // ATTENTION_HEADS
    num_heads = ATTENTION_HEADS


    # Linear projections
    Q = tf.matmul(inputs, attention_weights["W_q"])  # Shape: (batch_size, sequence_length, embedding_dim)
    K = tf.matmul(inputs, attention_weights["W_k"])  # Shape: (batch_size, sequence_length, embedding_dim)
    V = tf.matmul(inputs, attention_weights["W_v"])  # Shape: (batch_size, sequence_length, embedding_dim)
    # Possible experimentation you can try: Apply GELU activation function
    # Q = gelu(Q)
    # K = gelu(K)
    # V = gelu(V)

    # Split the embedding dimension into multiple heads
    Q = tf.reshape(Q, (-1, MAX_TOKEN_SIZE, num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)
    K = tf.reshape(K, (-1, MAX_TOKEN_SIZE, num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)
    V = tf.reshape(V, (-1, MAX_TOKEN_SIZE, num_heads, head_dim))  # Shape: (batch_size, sequence_length, num_heads, head_dim)# Compute the scaled dot-product attention for each head

    #Get the scores of how influential each token is to each other
    attention_scores = tf.einsum('bqhd,bkhd->bhqk', Q, K)  # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    attention_scores = attention_scores / tf.sqrt(tf.cast(head_dim, tf.float32))
    # Apply the stability fix by subtracting the max value
    max_attention_scores = tf.reduce_max(attention_scores, axis=-1, keepdims=True)
    attention_scores -= max_attention_scores

    # Apply the attention mask: set scores to a very large negative value where mask is 0 (pad tokens) to ignore them
    attention_mask = tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
    large_negative_value = -1e9
    attention_scores += attention_mask * large_negative_value

    #Apply softmax to get attention weights from scores, to then apply to the output
    #Manual softmax (may experiment with for computational efficiency): np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    score_weights = tf.nn.softmax(attention_scores, axis=-1)

    # Get attention weight adjustments across all heads based on the value matrix to apply to the output matrix to then add to each embedding input
    attention_output = tf.einsum('bhqk,bkhd->bqhd', score_weights, V)  # Shape: (batch_size, sequence_length, num_heads, head_dim)


    # Apply dropout to attention output (before residual connection)
    # Applied before reshaping (concatenating attention heads) to diversify neurons being used across each head, limiting reliance on a single neuron in a particular att_head

    attention_output = tf.nn.dropout(attention_output, rate=DROPOUT_RATE) if training else attention_output

    #Concatenate the heads to get
    attention_output = tf.reshape(attention_output, (-1, attention_output.shape[1], DIMENSION))  # Shape: (batch_size, sequence_length, embedding_dim)# Final linear layer


    # Shape: (batch_size, sequence_length, embedding_dim)
    output = tf.matmul(attention_output, attention_weights["W_o"])
    return output

# Unused: to maximize computational efficiency using Tensorflow method
# @tf.function
# def dropout(x, rate):
#     retain_prob = 1.0 - rate
#     mask = np.random.binomial(1, retain_prob, size=x.shape)
#     return x * mask / retain_prob