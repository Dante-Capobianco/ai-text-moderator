#Evaluate the data on the test data, offering a final check on its performance, beyond the evaluation on validation data in training/trainer.py

import tensorflow as tf

def evaluate_batch(correct_labels, predicted_labels):
    """
    Evaluate the number of correct labels out of 7 for each instance in the batch.

    Args:
    correct_labels (tf.Tensor): The ground truth labels (batch_size, 7).
    predicted_labels (tf.Tensor): The predicted labels (batch_size, 7).

    Returns:
    tf.Tensor: A tensor of shape (batch_size,) containing the number of correct labels for each instance.
    """
    # Convert probabilities to binary predictions (0 or 1) based on a threshold of 0.5
    binary_predictions = tf.cast(predicted_labels > 0.5, tf.int32)

    # Compare predictions to correct labels
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(correct_labels, binary_predictions), tf.int32), axis=1)

    return correct_predictions

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

def print_epoch_statistics(statistics):
    """
    Print the statistics for the epoch in the required format.

    Args:
    statistics (tf.Tensor): A tensor of shape (8,) containing the counts for each score (0/7, 1/7, ..., 7/7).
    """
    for i in range(8):  # 0 to 7 correct labels
        tf.print(f"Occurrence of {i}/7: ",statistics[i])
