#Code for word/positional vector embeddings of tokenized text input
#Create matrix of vocab size x embedding dimension, pass in array of tokenized text, output matrix of array size x embedding size
#Trainable
#Output is Bxmax lengthxembedding size
import time

from socialMediaClassificationTransformer.training.config import DIMENSION, MAX_TOKEN_SIZE, BATCH_SIZE, VOCAB_SIZE, \
    LEARNING_RATE, DROPOUT_RATE, INCLUDE_DYNAMIC_BALANCING, MASK_IN_EMBEDDINGS

import pickle
import tensorflow as tf

@tf.function
def _parse_function(proto):
    # Define your features to be parsed
    keys_to_features = {
        'token_ids': tf.io.FixedLenFeature([], tf.string),
        'attention_mask': tf.io.FixedLenFeature([], tf.string),
        'correct_labels': tf.io.FixedLenFeature([], tf.string),
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Parse the tensor from string
    parsed_features['token_ids'] = tf.io.parse_tensor(parsed_features['token_ids'], out_type=tf.int32)
    parsed_features['attention_mask'] = tf.io.parse_tensor(parsed_features['attention_mask'], out_type=tf.int32)
    parsed_features['correct_labels'] = tf.io.parse_tensor(parsed_features['correct_labels'], out_type=tf.int32)

    # Ensure correct_labels has the shape [7]
    parsed_features['correct_labels'] = tf.ensure_shape(parsed_features['correct_labels'], [7])

    return parsed_features['token_ids'], parsed_features['attention_mask'], parsed_features['correct_labels']

@tf.function
def compute_sample_weight(correct_labels, class_weights_tensor):

    # Calculate sample weights based on class labels and predefined class weights
    weights = tf.reduce_sum(tf.cast(correct_labels, tf.float32) * class_weights_tensor, axis=-1)
    return weights

@tf.function
def load_and_prepare_data(file_path, batch_size):
    # Load the data from the pickle file
    # data = load_data(file_path)

    # Removed for efficiency - using TF dataset
    # # Randomize the order of the data
    # random.shuffle(data)
    #
    # # Grab only the first 'batch_size' entries
    # batch_data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    #
    # return batch_data

    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    class_weights = [0.042985456201923594, 0.12078438894694984, 0.10910276845817135, 0.5187072652741072, 0.019151627059159584, 0.18837437842777963, 0.0008941156319089033]
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    """
    above calculate from the below computation:
    class_counts = {
        'spam': 34379,
        'toxic': 12235,
        'obscene': 13545,
        'threat': 2849,
        'insult': 77163,
        'identity_hate': 7845,
        'neutral': 1652803
    }

    total_samples = sum(class_counts.values())

    # Calculate normalized class weights
    class_weights = {label: total_samples / count for label, count in class_counts.items()}

    # Normalize to ensure the sum of weights equals 1
    normalized_class_weights = {label: weight / sum(class_weights.values()) for label, weight in class_weights.items()}

    # Output the normalized weights in the specified order
    weights_in_order = [
        normalized_class_weights['spam'],
        normalized_class_weights['toxic'],
        normalized_class_weights['obscene'],
        normalized_class_weights['threat'],
        normalized_class_weights['insult'],
        normalized_class_weights['identity_hate'],
        normalized_class_weights['neutral']
    ]

    print(weights_in_order)
    """
    def sample_weighted_examples(token_ids, attention_mask, correct_labels):
        sample_weight = compute_sample_weight(correct_labels, class_weights_tensor)

        return token_ids, attention_mask, correct_labels, sample_weight

    if INCLUDE_DYNAMIC_BALANCING:
        dataset = dataset.map(sample_weighted_examples, num_parallel_calls=tf.data.AUTOTUNE)
        # Resample the dataset to balance classes
        dataset = tf.data.Dataset.sample_from_datasets(
            [dataset],
            weights=dataset.map(lambda token_ids, attention_mask, correct_labels, sample_weight: sample_weight),
            seed=42,
            stop_on_empty_dataset=True
        )
        # Discard the sample_weight and return only the necessary components
        dataset = dataset.map(lambda token_ids, attention_mask, correct_labels, sample_weight: (
        token_ids, attention_mask, correct_labels))
    #Less computationally demanding method to attempt to balance class representation to reduce overfitting & avoid order
    # dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset




# @tf.function
# def load_data(file_path):
#     with open(file_path, 'rb') as file:
#         try:
#             data = pickle.load(file)
#             for entry in data:
#                 print(f"Yielding: {entry}")
#                 yield entry
#         except EOFError as e:
#             raise EOFError(f"Error loading {file_path}: {e}")

@tf.function
def get_positional_encoding(seq_len, d_model):
    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))

    # Apply sine to even indices
    sin_term = tf.sin(position * div_term)

    # Apply cosine to odd indices
    cos_term = tf.cos(position * div_term)

    # Interleave sin and cos terms
    pe = tf.concat([sin_term, cos_term], axis=-1)

    # Ensure that the pe tensor has the right shape if d_model is odd
    pe = tf.reshape(pe, (seq_len, d_model))

    return pe
@tf.function
def embedding_lookup(token_ids_batch, embedding_weights):
    return tf.nn.embedding_lookup(embedding_weights, token_ids_batch)
@tf.function
def forward_pass(token_ids_batch, embedding_weights, attention_mask, training=True):

    # Lookup embeddings
    embeddings = embedding_lookup(token_ids_batch, embedding_weights[0]["embed_weights"])

    sum_embeddings = embeddings + get_positional_encoding(MAX_TOKEN_SIZE, DIMENSION)

    if MASK_IN_EMBEDDINGS:
        sum_embeddings *= tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=-1)

    sum_embeddings = tf.nn.dropout(sum_embeddings, rate=DROPOUT_RATE) if training else sum_embeddings



    return sum_embeddings

def init_embedding_weights():
    embed_weights = []
    #Consider He initialization; currently using xavier
    weights = {
            "embed_weights": tf.Variable(initial_value=tf.random.uniform(shape=(VOCAB_SIZE, DIMENSION), minval=-0.5, maxval=0.5, dtype=tf.float32),
                                         trainable=True  # Indicates that the variable should be trainable

                )
    }
    embed_weights.append(weights)
    return embed_weights
@tf.function
def init_embedding(file_path):
    try:
        return load_and_prepare_data(file_path, BATCH_SIZE)

    except Exception as e:
        print(f"Failed to process: {e}")

# Unused for efficiency
# @tf.function
# def dropout(x, rate):
#     retain_prob = 1.0 - rate
#     mask = np.random.binomial(1, retain_prob, size=x.shape)
#     return x * mask / retain_prob