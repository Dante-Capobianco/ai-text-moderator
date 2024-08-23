#Training loop, including batching, loss calculation, backpropagation using the optimizer to update weights, evaluating the model on validation data every 1+ epochs
#Saves model as well
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras.losses import BinaryCrossentropy

from socialMediaClassificationTransformer.models.classification_head import init_classification_weights

from socialMediaClassificationTransformer.models.embedding import init_embedding_weights, init_embedding
from socialMediaClassificationTransformer.models.transformer import run_transformer
from socialMediaClassificationTransformer.models.attention import init_attention_weights, xavier_initialization
from socialMediaClassificationTransformer.models.feedforward import init_ffn_weights
from socialMediaClassificationTransformer.scripts.evaluate import evaluate_batch, update_epoch_statistics, \
    print_epoch_statistics
from socialMediaClassificationTransformer.training.config import EPOCHS, DIMENSION, MAX_TOKEN_SIZE, BATCH_SIZE, \
    LEARNING_RATE, WARMUP_STEPS, DECAY_STEPS, ADAMW_WEIGHT_DECAY, DECAY_RATE, \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY, VOCAB_SIZE
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers.experimental import AdamW

from socialMediaClassificationTransformer.training.trainer import WarmupThenDecaySchedule, loss_with_neutral_penalty, \
    flatten_weights

# Set global mixed precision policy - only include if computer cannot handle computations
set_global_policy('float16')

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def save_model_checkpoint(checkpoint_manager):
    """Saves the model checkpoint using the checkpoint manager."""
    if checkpoint_manager is not None:
        save_path = checkpoint_manager.save()
        tf.print("Model checkpoint saved at:", save_path)
    else:
        raise ValueError("CheckpointManager is not initialized properly.")

@tf.function
def train(data, epochs, embed_weights, attention_weights, ffn_weights, classification_weights, optimizer, loss_fn, epoch_statistics):
    for epoch in tf.range(epochs):  # Simulate the training epochs
        tf.print("Epoch", epoch + 1, "/", epochs)
        i = 1

        for token_ids_batch, attention_masks_batch, correct_labels_batch in data:
            tf.print("Batch", i)
            all_weights = flatten_weights(embed_weights, attention_weights, ffn_weights, classification_weights)

            # Ensure the shape of token_ids_batch and attention_masks_batch
            token_ids_batch = tf.ensure_shape(token_ids_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])
            attention_masks_batch = tf.ensure_shape(attention_masks_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])

            #call transformer to run through layers once
            #token_ids_batch (list of tensors each containing list of numbers representing tokens)
            #3 checks for transformer output: stability (highs & lows consistent across batches, spread in values
            # decrease over epochs, not overly narrow variance from layer norm
            with tf.GradientTape() as tape:
                output = run_transformer(token_ids_batch, embed_weights, attention_weights, attention_masks_batch, True, ffn_weights, classification_weights)
                tf.print(output)
                # Calculate loss
                if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:
                    loss_value = loss_fn(tf.cast(correct_labels_batch, tf.float32), tf.cast(output, tf.float32))
                    probabilities = output
                else:
                    # Compute sigmoid cross-entropy loss directly
                    loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(correct_labels_batch, tf.float32), logits=output)
                    loss_value = tf.reduce_mean(loss_value)
                    probabilities = tf.nn.sigmoid(output)
                loss_value = loss_with_neutral_penalty(loss_value, probabilities)


            # Backpropagation: compute gradients
            grads = tape.gradient(loss_value, all_weights)

            # # Optimization: update weights
            optimizer.apply_gradients(zip(grads, all_weights))

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
    data_items = init_embedding("../data/tokenized/tokenized_train.tfrecord")
    #data_items = data_items.apply(tf.data.experimental.prefetch_to_device('/GPU:0')) - if needed to ensure GPU loads dataset

    # Initialize statistics tensor to track correct prediction counts (0 to 7 correct labels)
    epoch_statistics = tf.zeros([8], dtype=tf.int32)



    # Instantiate the AdamW optimizer
    optimizer = AdamW(
        learning_rate=WarmupThenDecaySchedule(LEARNING_RATE, WARMUP_STEPS, DECAY_STEPS, DECAY_RATE ),   # Set your learning rate
        weight_decay=ADAMW_WEIGHT_DECAY     # Set your weight decay
    )

    if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:

        # Define the loss function for binary classification
        loss_fn = BinaryCrossentropy(from_logits=True, label_smoothing=0.0)
    else:
        loss_fn = None

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, weights=flatten_weights(embedding_weights, att_weights, ffn_weight, class_weights))


    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)



    # Start training
    train(data_items, EPOCHS, embedding_weights, att_weights, ffn_weight, class_weights, optimizer, loss_fn, epoch_statistics)

    # Save the model at the end of each training session
    save_model_checkpoint(checkpoint_manager)
