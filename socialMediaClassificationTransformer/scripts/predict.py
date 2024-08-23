#Use loaded model to predict label on test - ability to redesign to use on new, real-world data, either live user input or new datasets
#Only one final possible modification - BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY
# with SPAM_THRESHOLD, TOXIC_THRESHOLD, OBSCENE_THRESHOLD, THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD
# for further finetuning on classifying the output
from socialMediaClassificationTransformer.models.classification_head import init_classification_weights

from socialMediaClassificationTransformer.models.embedding import init_embedding_weights, init_embedding
from socialMediaClassificationTransformer.models.transformer import run_transformer
from socialMediaClassificationTransformer.models.attention import init_attention_weights
from socialMediaClassificationTransformer.models.feedforward import init_ffn_weights
from socialMediaClassificationTransformer.scripts.evaluate import evaluate_batch, update_epoch_statistics, \
    print_epoch_statistics
from socialMediaClassificationTransformer.training.config import MAX_TOKEN_SIZE, BATCH_SIZE, \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

from socialMediaClassificationTransformer.training.trainer import WarmupThenDecaySchedule, loss_with_neutral_penalty, \
    flatten_weights

# Set global mixed precision policy - only include if computer cannot handle computations
set_global_policy('mixed_float16')

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

@tf.function
def test(data, embed_weights, attention_weights, ffn_weights, classification_weights):
    i = 1


    # Initialize statistics tensor to track correct prediction counts (0 to 7 correct labels)
    epoch_statistics = tf.Variable(tf.zeros([8], dtype=tf.int32))

    for token_ids_batch, attention_masks_batch, correct_labels_batch in data:
        tf.print("Batch", i)

        # Ensure the shape of token_ids_batch and attention_masks_batch
        token_ids_batch = tf.ensure_shape(token_ids_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])
        attention_masks_batch = tf.ensure_shape(attention_masks_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])

        #call transformer to run through layers once
        output = run_transformer(token_ids_batch, embed_weights, attention_weights, attention_masks_batch, False, ffn_weights, classification_weights)

        if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:
            probabilities = output
        else:
            probabilities = tf.nn.sigmoid(output)

        # Evaluate how many labels were correctly predicted
        correct_predictions = evaluate_batch(correct_labels_batch, probabilities)
        epoch_statistics = update_epoch_statistics(correct_predictions, epoch_statistics)

        i += 1

    # Print the statistics for the epoch
    print_epoch_statistics(epoch_statistics)


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


    # Start training
    test(data_items, embedding_weights, att_weights, ffn_weight, class_weights)

