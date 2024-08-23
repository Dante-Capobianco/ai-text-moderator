#Developing architecture, combining each component in the structure

import tensorflow as tf

from socialMediaClassificationTransformer.models.classification_head import classify_sequences
from socialMediaClassificationTransformer.models.embedding import forward_pass
from socialMediaClassificationTransformer.models.attention import multi_head_self_attention
from socialMediaClassificationTransformer.training.config import MAX_TOKEN_SIZE, DIMENSION, BATCH_SIZE, NUM_LAYERS, \
    INCLUDE_CLS
from socialMediaClassificationTransformer.models.feedforward import feedforward_network

# Initialize the LayerNormalization outside the @tf.function
layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
@tf.function
def run_transformer(token_ids, embedding_weights, attention_weights, attention_masks, training, ffn_weights, classification_weights):
    # Forward pass through embedding layer
    embedding_output = forward_pass(token_ids, embedding_weights, attention_masks, training)

    for layer in range(NUM_LAYERS):
        # Extract the weights for the current layer
        current_attention_weights = attention_weights[layer]
        current_ffn_weights = ffn_weights[layer]

        # Apply multi-head self-attention mechanism
        attention_output = multi_head_self_attention(embedding_output, current_attention_weights, attention_masks, training)

        # Add residual connection: combine attention output with the original embedding output
        residual_output = attention_output + embedding_output

        # Ensure the shape of residual_output - to define shape during graph creation
        residual_output = tf.ensure_shape(residual_output, [BATCH_SIZE, MAX_TOKEN_SIZE, DIMENSION])

        normalized_output = layer_norm(residual_output)

        # Feedforward network with residual connection
        ffn_output = feedforward_network(normalized_output, current_ffn_weights[0], current_ffn_weights[1], training)

        # Another residual connection after the FNN
        residual_output = normalized_output + ffn_output

        #Normalize residually connected output
        embedding_output = layer_norm(residual_output)


    #Use CLS token to classify
    if not INCLUDE_CLS:
        embedding_output = embedding_output[:, 1:, :]


    return classify_sequences(embedding_output, classification_weights)


# adding residual connections & layer normalization after FNN, and before final classification head

#optimizations: use gradient checkpointing: from tensorflow.python.keras.utils import tf_utils
# tf_utils.enable_gradient_checkpointing(model)
# possibly delete and clear memory of large tensors avoid operations not supporting gpu, increase batch size, possibly delete unneeded var like old inputs in the transformer, ensure using gpu instead of cpu, ensure parallel processing, switch to tensorflow functions (especially dropout function), don't shuffle data, lower dimensionality of embedding,
#add logging to see where most time spent, use python profiler like cprofile or line_profiler, monitor memory usage for memory leak (possibly  delete embedding & attention output after not needed,
#increase heap size: -Xms512m -Xmx4096m in idea64.exe.vmoptions OR File > Settings > Build, Execution, Deployment > Compiler > Shared build process heap size and increase the memory allocation to ensure your project has enough resources (do not allocate too much b/c starve rest of computer)

