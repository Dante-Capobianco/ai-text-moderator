#Loss function

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import float32

from socialMediaClassificationTransformer.training.config import \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY, PENALTY_FACTOR

class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    @tf.function
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, decay_rate):
        super(WarmupThenDecaySchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        self.warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        self.decay_steps = tf.cast(self.decay_steps, tf.float32)
        self.decay_rate = tf.cast(self.decay_rate, tf.float32)
        self.initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)
        # Warmup phase
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step / self.warmup_steps)
        # Decay phase
        return self.initial_learning_rate * self.decay_rate ** ((step - self.warmup_steps) / self.decay_steps)

@tf.function
def loss_with_neutral_penalty(loss, probabilities):
    # Implement the penalty for neutral with other labels
    neutral_predictions = probabilities[:, -1]  # Assuming neutral is the last label
    other_predictions = probabilities[:, :-1]

    # Calculate the penalty: if neutral is predicted with high probability and any other label is predicted with high probability
    penalty = tf.reduce_sum(neutral_predictions * tf.reduce_max(other_predictions, axis=1))

    # Combine the loss and penalty
    total_loss = loss + (tf.cast(penalty, float32) * PENALTY_FACTOR)
    return total_loss


def flatten_weights(embed_weights, attention_weights, ffn_weights, classification_weights):
    # Flatten embed_weights (already a single tensor)
    flattened_weights = [embed_weights[0]["embed_weights"]]

    # Flatten attention_weights (list of dictionaries)
    for layer_weights in attention_weights:
        flattened_weights.extend(layer_weights.values())

    # Flatten ffn_weights (list of tuples)
    for W_1, W_2 in ffn_weights:
        flattened_weights.append(W_1)
        flattened_weights.append(W_2)

    # Flatten classification_weights (already a single tensor)
    flattened_weights.append(classification_weights[0]["class_weights"])

    return flattened_weights