#Evaluate the data on the data, offering a final check on its performance

import tensorflow as tf

from socialMediaClassificationTransformer.training.config import SPAM_THRESHOLD, TOXIC_THRESHOLD, OBSCENE_THRESHOLD, \
    THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD, TEST_SPAM_THRESHOLD, \
    TEST_THREAT_THRESHOLD, TEST_INSULT_THRESHOLD, TEST_IDENTITY_HATE_THRESHOLD, TEST_NEUTRAL_THRESHOLD, \
    TEST_TOXIC_THRESHOLD, TEST_OBSCENE_THRESHOLD


@tf.function
def evaluate_batch(correct_labels, predicted_labels, training):
    """
    Evaluate the number of correct labels out of 7 for each instance in the batch.

    Args:
    correct_labels (tf.Tensor): The ground truth labels (batch_size, 7).
    predicted_labels (tf.Tensor): The predicted labels (batch_size, 7).

    Returns:
    tf.Tensor: A tensor of shape (batch_size,) containing the number of correct labels for each instance.
    """
    # Convert probabilities to binary predictions (0 or 1) based on a threshold of 0.5
    if training:
        thresholds = tf.constant([SPAM_THRESHOLD, TOXIC_THRESHOLD, OBSCENE_THRESHOLD, THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD], dtype=tf.float32)
    else:
        thresholds = tf.constant([TEST_SPAM_THRESHOLD, TEST_TOXIC_THRESHOLD, TEST_OBSCENE_THRESHOLD, TEST_THREAT_THRESHOLD, TEST_INSULT_THRESHOLD, TEST_IDENTITY_HATE_THRESHOLD, TEST_NEUTRAL_THRESHOLD], dtype=tf.float32)
    binary_predictions = tf.cast(predicted_labels > thresholds, tf.int32)

    # Compare predictions to correct labels
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(correct_labels, binary_predictions), tf.int32), axis=1)

    return correct_predictions
@tf.function
def update_epoch_statistics(correct_predictions, statistics):
    """
    Update the epoch statistics with the results from a batch.

    Args:
    correct_predictions (tf.Tensor): A tensor of shape (batch_size,) containing the number of correct labels for each instance.
    statistics (tf.Tensor): A tensor of shape (8,) containing the counts for each score (0/7, 1/7, ..., 7/7).

    Returns:
    tf.Tensor: Updated statistics tensor.
    """

    # Update the statistics tensor
    for i in range(8):  # 0 to 7 correct labels
        # Calculate the number of instances where the correct prediction count equals i
        count_i = tf.reduce_sum(tf.cast(tf.equal(correct_predictions, i), tf.int32))
        # Update the statistics tensor using tensor_scatter_nd_add
        indices = tf.constant([[i]])
        statistics = tf.tensor_scatter_nd_add(statistics, indices, tf.expand_dims(count_i, 0))

    return statistics
@tf.function
def print_epoch_statistics(statistics):
    """
    Print the statistics for the epoch in the required format.

    Args:
    statistics (tf.Tensor): A tensor of shape (8,) containing the counts for each score (0/7, 1/7, ..., 7/7).
    """
    for i in range(8):  # 0 to 7 correct labels
        tf.print(f"Occurrence of {i}/7: ",statistics[i])

@tf.function
def calculate_average_accuracy(statistics):
    """
    Calculate and print the average accuracy for the epoch based on the occurrence of correct labels.

    Args:
    statistics (tf.Tensor): A tensor of shape (8,) containing the counts for each score (0/7, 1/7, ..., 7/7).

    Returns:
    tf.Tensor: The average accuracy as a percentage.
    """
    total_instances = tf.reduce_sum(statistics)  # Total number of instances in the epoch
    weighted_sum = tf.constant(0.0, dtype=tf.float32)

    for i in range(8):  # 0 to 7 correct labels
        weighted_sum += tf.cast(i, tf.float32) * tf.cast(statistics[i], tf.float32)

    average_correctness = weighted_sum / (7.0 * tf.cast(total_instances, tf.float32))  # Divide by 7 to normalize to a percentage
    average_accuracy = average_correctness * 100.0  # Convert to percentage
    return average_accuracy

