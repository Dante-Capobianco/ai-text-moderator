#Developing architecture, combining each component in the structure

import tensorflow as tf
from socialMediaClassificationTransformer.models.embedding import forward_pass
from socialMediaClassificationTransformer.models.attention import multi_head_self_attention
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def run_transformer(embeddings, embedding_weights, attention_weights, training):
    # Forward pass through embedding layer
    embedding_output = forward_pass(embeddings, embedding_weights, training)

    # Apply multi-head self-attention mechanism
    attention_output = multi_head_self_attention(embedding_output, attention_weights, training)

    # Add residual connection: combine attention output with the original embedding output
    residual_output = attention_output + embedding_output

    # Apply Layer Normalization after the residual connection
    layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
    normalized_output = layer_norm(residual_output)


# adding residual connections & layer normalization after FNN, and before final classification head

#optimizations: switch to tensorflow gpu & test nvidia-smi, avoid operations not supporting gpu, get main gpu as nvidia not intel, increase batch size, possibly delete unneeded var like old inputs in the transformer, ensure using gpu instead of cpu, ensure parallel processing, switch to tensorflow functions (especially dropout function), don't shuffle data, lower dimensionality of embedding,
#add logging to see where most time spent, use python profiler like cprofile or line_profiler, monitor memory usage for memory leak (possibly  delete embedding & attention output after not needed,
#increase heap size: -Xms512m -Xmx4096m in idea64.exe.vmoptions OR File > Settings > Build, Execution, Deployment > Compiler > Shared build process heap size and increase the memory allocation to ensure your project has enough resources (do not allocate too much b/c starve rest of computer)

