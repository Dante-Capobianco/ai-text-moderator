from keras.losses import BinaryCrossentropy

from socialMediaClassificationTransformer.data.data_loader import clean_text
from socialMediaClassificationTransformer.models.classification_head import init_classification_weights

from socialMediaClassificationTransformer.models.embedding import init_embedding_weights, init_embedding
from socialMediaClassificationTransformer.models.tokenizer import load_vocab, tokenize_dataset
from socialMediaClassificationTransformer.models.transformer import run_transformer
from socialMediaClassificationTransformer.models.attention import init_attention_weights
from socialMediaClassificationTransformer.models.feedforward import init_ffn_weights
from socialMediaClassificationTransformer.training import config
from socialMediaClassificationTransformer.training.config import MAX_TOKEN_SIZE, BATCH_SIZE, \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY, TOTAL_TESTING_BATCHES, TEST_SPAM_THRESHOLD, \
    TEST_TOXIC_THRESHOLD, TEST_OBSCENE_THRESHOLD, TEST_THREAT_THRESHOLD, TEST_INSULT_THRESHOLD, \
    TEST_IDENTITY_HATE_THRESHOLD, TEST_NEUTRAL_THRESHOLD
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from socialMediaClassificationTransformer.training.trainer import WarmupThenDecaySchedule, loss_with_neutral_penalty, \
    flatten_weights, f1_score

# Set global mixed precision policy - only include if computer cannot handle computations
set_global_policy('mixed_float16')

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


if __name__ == "__main__":
    #Use GPU with memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled")
        except RuntimeError as e:
            print(e)

    #initialize weights & vocab
    embedding_weights = init_embedding_weights()
    att_weights = init_attention_weights()
    ffn_weight = init_ffn_weights()
    class_weights = init_classification_weights()
    vocab = load_vocab('../data/vocab/vocab.txt')

    # Restore the weights from the checkpoint
    checkpoint = tf.train.Checkpoint(weights=flatten_weights(embedding_weights, att_weights, ffn_weight, class_weights))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=5)

    # Restore the weights
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    print("Model weights restored from:", checkpoint_manager.latest_checkpoint)

    #Get user input
    inputs = input("Enter text (max. approx. 500 words) to have it analyzed: ")

    cleaned_input = clean_text(inputs)

    # Prepare the input data as a single-entry list of dictionaries to simulate the dataset structure
    input_data = [{'text': cleaned_input, 'labels': {}}]  # Empty labels since they aren't used in this context

    # Tokenize the input data
    tokenized_input = tokenize_dataset(input_data, vocab)

    # Convert to Tensor and ensure the shape is (1, MAX_TOKEN_SIZE)
    token_ids = tf.constant([tokenized_input[0]['token_ids']], dtype=tf.int32)
    attention_mask = tf.constant([tokenized_input[0]['attention_mask']], dtype=tf.int32)
    token_ids = tf.ensure_shape(token_ids, [1, MAX_TOKEN_SIZE])
    attention_mask = tf.ensure_shape(attention_mask, [1, MAX_TOKEN_SIZE])

    # Run the input through the transformer model
    output = run_transformer(token_ids, embedding_weights, att_weights, attention_mask, False, ffn_weight, class_weights, True)

    #Get the probabilities to then apply the threshold to finally get the final answer for each label
    if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:
        probabilities = output
    else:
        probabilities = tf.nn.sigmoid(output)

    thresholds = tf.constant([TEST_SPAM_THRESHOLD, TEST_TOXIC_THRESHOLD, TEST_OBSCENE_THRESHOLD, TEST_THREAT_THRESHOLD, TEST_INSULT_THRESHOLD, TEST_IDENTITY_HATE_THRESHOLD, TEST_NEUTRAL_THRESHOLD], dtype=tf.float32)
    binary_predictions = tf.cast(probabilities > thresholds, tf.int32)

    labels = ["Spam", "Toxic", "Obscene", "Threat", "Insult", "Identity Hate", "Neutral"]

    # Iterate over each label and its corresponding prediction
    for i in range(7):
        if i == 6:  # This is the index for "Neutral"
            if all(binary_predictions[0, :6] == 0):  # Check if all previous predictions are 0
                print(f"{labels[i]}: True")
            else:
                print(f"{labels[i]}: False")
        else:
            if binary_predictions[0, i] == 1:
                print(f"{labels[i]}: True")
            else:
                print(f"{labels[i]}: False")
