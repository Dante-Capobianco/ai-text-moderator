#Training loop, including batching, loss calculation, backpropagation using the optimizer to update weights, evaluating the model on validation data every 1+ epochs
#Saves model as well

#APPROACH: Start off with low step counts to experiment with various different parameter tunings
#Once have a strong tuning, increase step count & hypertune parameters to maximize model
#Once have top tunings, experiment with step count to optimize training without overfitting

import tensorflow as tf
from keras.optimizers import Adam

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
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY, VOCAB_SIZE, TOTAL_VALIDATION_BATCHES, PATIENCE, \
    ACCEPTABLE_LOSS_GAP
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers.experimental import AdamW

from socialMediaClassificationTransformer.training.trainer import WarmupThenDecaySchedule, loss_with_neutral_penalty, \
    flatten_weights, f1_score

# Set global mixed precision policy - only include if computer cannot handle computations
set_global_policy('float16')

import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

@tf.function
def train(data, epochs, embed_weights, attention_weights, ffn_weights, classification_weights, optimizer, loss_fn, epoch_statistics, validation_data, checkpoint_manage):
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = 0
    wait = 0
    for epoch in tf.range(epochs):  # Simulate the training epochs
        tf.print("Epoch", epoch + 1, "/", epochs)
        training_steps = 1
        total_training_loss = 0.0

        for token_ids_batch, attention_masks_batch, correct_labels_batch in data:
            #Reset to beginning of the dataset for the next epoch after finished with all steps initialized in config.py
            if training_steps >= WARMUP_STEPS + DECAY_STEPS:
                break
            all_weights = flatten_weights(embed_weights, attention_weights, ffn_weights, classification_weights)


            tf.print("Batch", training_steps)
            # Ensure the shape of token_ids_batch and attention_masks_batch
            token_ids_batch = tf.ensure_shape(token_ids_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])
            attention_masks_batch = tf.ensure_shape(attention_masks_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])

            #call transformer to run through layers once
            #token_ids_batch (list of tensors each containing list of numbers representing tokens)
            #3 checks for transformer output: stability (highs & lows consistent across batches, spread in values
            # decrease over epochs, not overly narrow variance from layer norm
            with tf.GradientTape() as tape:
                output = run_transformer(token_ids_batch, embed_weights, attention_weights, attention_masks_batch, True, ffn_weights, classification_weights)
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


            optimizer.apply_gradients(zip(grads, all_weights))

            # Evaluate how many labels were correctly predicted
            total_training_loss += loss_value
            correct_predictions = evaluate_batch(correct_labels_batch, probabilities)
            epoch_statistics = update_epoch_statistics(correct_predictions, epoch_statistics)

            training_steps += 1


            # Implemented due to GPU limitations when applying above method
            #NEED TO USE DECAY RATE OR USE ADAM
            # for var, grad in zip(all_weights, grads):
            #     if grad is not None:
            #         var.assign_sub(grad * optimizer.learning_rate)

            # Can work with this if Adam optimizer is too computationally expensive - manual version of Adam (will require set up of its variables)
            # for j, (var, grad) in enumerate(zip(all_weights, grads)):
            #     if grad is not None:
            #         m[j] = beta1 * m[j] + (1 - beta1) * grad
            #         v[j] = beta2 * v[j] + (1 - beta2) * tf.square(grad)
            #
            #         m_hat = m[j] / (1 - tf.pow(beta1, t))
            #         v_hat = v[j] / (1 - tf.pow(beta2, t))
            #
            #         var.assign_sub(LEARNING_RATE * m_hat / (tf.sqrt(v_hat) + epsilon))
            #
            # t += 1
            # i += 1

        avg_training_loss = total_training_loss / tf.cast(training_steps, tf.float32)
        tf.print("Training Loss:", avg_training_loss)

        # Validation Phase
        total_val_loss = 0.0
        total_val_f1 = 0.0
        val_steps = 0

        tf.print("Validation for Epoch", epoch + 1, "/", epochs)
        for val_token_ids_batch, val_attention_masks_batch, val_correct_labels_batch in validation_data:
            if val_steps >= TOTAL_VALIDATION_BATCHES:
                break
            val_token_ids_batch = tf.ensure_shape(val_token_ids_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])
            val_attention_masks_batch = tf.ensure_shape(val_attention_masks_batch, [BATCH_SIZE, MAX_TOKEN_SIZE])

            output = run_transformer(val_token_ids_batch, embed_weights, attention_weights, val_attention_masks_batch, False, ffn_weights, classification_weights)

            if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:
                val_loss_value = loss_fn(tf.cast(val_correct_labels_batch, tf.float32), tf.cast(output, tf.float32))
                val_probabilities = output
            else:
                val_loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(val_correct_labels_batch, tf.float32), logits=output)
                val_loss_value = tf.reduce_mean(val_loss_value)
                val_probabilities = tf.nn.sigmoid(output)

            val_f1 = f1_score(val_correct_labels_batch, val_probabilities)
            total_val_loss += val_loss_value
            total_val_f1 += val_f1
            val_steps += 1

        avg_val_loss = total_val_loss / tf.cast(val_steps, tf.float32)
        avg_val_f1 = total_val_f1 / tf.cast(val_steps, tf.float32)
        tf.print("Validation Loss:", avg_val_loss)
        tf.print("Validation F1:", avg_val_f1)

        # Early Stopping Logic - ensure validation loss & accuracy improves, and training loss does not significantly
        # gap validation loss (indicating overfitting)
        if avg_val_loss < best_val_loss and avg_val_f1 > best_val_f1 and avg_val_loss <= avg_training_loss + ACCEPTABLE_LOSS_GAP:
            tf.print("Best epoch to date")
            best_val_loss = avg_val_loss
            best_val_f1 = avg_val_f1
            best_epoch = epoch
            wait = 0  # Reset the patience counter
            # Save checkpoint outside the graph
            tf.numpy_function(lambda: checkpoint_manage.save(), [], [])
        else:
            tf.print("This epoch did not improve from the previous")
            wait += 1
            if wait >= PATIENCE:
                tf.print(f"Early stopping at epoch {epoch + 1} with best epoch at epoch {best_epoch}")
                break

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
    val_items = init_embedding("../data/tokenized/tokenized_validation.tfrecord")
    #data_items = data_items.apply(tf.data.experimental.prefetch_to_device('/GPU:0')) - if needed to ensure GPU loads dataset

    # Initialize statistics tensor to track correct prediction counts (0 to 7 correct labels)
    statistics = tf.zeros([8], dtype=tf.int32)

    # Instantiate the AdamW optimizer
    #Unused due to GPU limitations - if enough computer power, use this
    # optimizer = AdamW(
    #     learning_rate=WarmupThenDecaySchedule(LEARNING_RATE, WARMUP_STEPS, DECAY_STEPS, DECAY_RATE ),   # Set your learning rate
    #     weight_decay=ADAMW_WEIGHT_DECAY     # Set your weight decay
    # )
    optimizer = Adam(learning_rate=WarmupThenDecaySchedule(LEARNING_RATE, WARMUP_STEPS, DECAY_STEPS, DECAY_RATE ))

    if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:

        # Define the loss function for binary classification
        loss_fn = BinaryCrossentropy(from_logits=True, label_smoothing=0.0)
    else:
        loss_fn = None

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, weights=flatten_weights(embedding_weights, att_weights, ffn_weight, class_weights))


    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=3)



    # Start training
    train(data_items, EPOCHS, embedding_weights, att_weights, ffn_weight, class_weights, optimizer, loss_fn, statistics, val_items, checkpoint_manager)

    # Save the model at the end of each training session
    #save_model_checkpoint(checkpoint_manager)
