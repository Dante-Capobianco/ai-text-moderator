#Use loaded model to predict label on test - ability to redesign to use on new, real-world data, either live user input or new datasets
#Only one final possible modification - BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY
# with SPAM_THRESHOLD, TOXIC_THRESHOLD, OBSCENE_THRESHOLD, THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD
# for further finetuning on classifying the output
from keras.losses import BinaryCrossentropy

from socialMediaClassificationTransformer.models.classification_head import init_classification_weights

from socialMediaClassificationTransformer.models.embedding import init_embedding_weights, init_embedding
from socialMediaClassificationTransformer.models.transformer import run_transformer
from socialMediaClassificationTransformer.models.attention import init_attention_weights
from socialMediaClassificationTransformer.models.feedforward import init_ffn_weights
from socialMediaClassificationTransformer.scripts.evaluate import evaluate_batch, update_epoch_statistics, \
    print_epoch_statistics
from socialMediaClassificationTransformer.training.config import MAX_TOKEN_SIZE, BATCH_SIZE, \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY, TOTAL_TESTING_BATCHES
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from socialMediaClassificationTransformer.training.trainer import WarmupThenDecaySchedule, loss_with_neutral_penalty, \
    flatten_weights, f1_score

# Set global mixed precision policy - only include if computer cannot handle computations
set_global_policy('mixed_float16')

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

@tf.function
def test(data, embed_weights, attention_weights, ffn_weights, classification_weights, loss_fn):
    i = 1
    total_testing_loss = 0.0
    total_testing_f1 = 0.0


    # Initialize statistics tensor to track correct prediction counts (0 to 7 correct labels)
    epoch_statistics = tf.Variable(tf.zeros([8], dtype=tf.int32))

    for token_ids_batch, attention_masks_batch, correct_labels_batch in data:
        if i >= TOTAL_TESTING_BATCHES:
            break
        tf.print("Batch", i)

        # Ensure the shape of token_ids_batch and attention_masks_batch
        token_ids_batch = tf.ensure_shape(token_ids_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])
        attention_masks_batch = tf.ensure_shape(attention_masks_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])

        #call transformer to run through layers once
        output = run_transformer(token_ids_batch, embed_weights, attention_weights, attention_masks_batch, False, ffn_weights, classification_weights)

        if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:
            loss_value = loss_fn(tf.cast(correct_labels_batch, tf.float32), tf.cast(output, tf.float32))
            probabilities = output
        else:
            # Compute sigmoid cross-entropy loss directly
            loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(correct_labels_batch, tf.float32), logits=output)
            loss_value = tf.reduce_mean(loss_value)
            probabilities = tf.nn.sigmoid(output)
        loss_value = loss_with_neutral_penalty(loss_value, probabilities)
        total_testing_loss += loss_value
        total_testing_f1 += f1_score(correct_labels_batch, probabilities)

        # Evaluate how many labels were correctly predicted
        correct_predictions = evaluate_batch(correct_labels_batch, probabilities)
        epoch_statistics = update_epoch_statistics(correct_predictions, epoch_statistics)

        i += 1

    # Print the statistics for the epoch
    print_epoch_statistics(epoch_statistics)
    avg_testing_loss = total_testing_loss / tf.cast(i, tf.float32)
    avg_test_f1 = total_testing_f1 / tf.cast(i, tf.float32)
    tf.print("Testing Loss:", avg_testing_loss)
    tf.print("Testing F1:", avg_test_f1)


# Example usage
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled")
        except RuntimeError as e:
            print(e)

    embedding_weights = init_embedding_weights()
    att_weights = init_attention_weights()
    ffn_weight = init_ffn_weights()
    class_weights = init_classification_weights()

    # Restore the weights from the checkpoint
    checkpoint = tf.train.Checkpoint(weights=flatten_weights(embedding_weights, att_weights, ffn_weight, class_weights))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)

    # Restore the weights
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    print("Model weights restored from:", checkpoint_manager.latest_checkpoint)


    data_items = init_embedding("../data/tokenized/tokenized_test.tfrecord")

    if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:

        # Define the loss function for binary classification
        loss_fn = BinaryCrossentropy(from_logits=True, label_smoothing=0.0)
    else:
        loss_fn = None


    # Start training
    test(data_items, embedding_weights, att_weights, ffn_weight, class_weights, loss_fn)

