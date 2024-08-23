#Creates classification/labelling of transformer output using a linear layer followed by softmax
#if from 0 to 1 scale, a label has a rating > 0.5, consider it to match that label
#incorrect if both neutral & a negative label > 0.5

#

#input in form of [batch_size, embedding_dim]
import tensorflow as tf

from socialMediaClassificationTransformer.training.config import DIMENSION, SPAM_THRESHOLD, TOXIC_THRESHOLD, \
    OBSCENE_THRESHOLD, THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD, \
    BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY
from socialMediaClassificationTransformer.models.attention import xavier_initialization

#Can also attempt CLS classification with the INCLUDE_CLS parameter set to True; or avg_pooling


def init_classification_weights():
    class_weights = []
    #Consider He initialization; currently using xavier
    weights = {
        "class_weights": tf.Variable(initial_value=xavier_initialization((DIMENSION, 7)),
                                     trainable=True  # Indicates that the variable should be trainable

                                     )
    }
    class_weights.append(weights)
    return class_weights

@tf.function
def max_pooling(input_tensor):
    """
    Apply max pooling over the sequence length dimension.

    Args:
    - input_tensor (tf.Tensor): A 3D tensor with shape [batch_size, seq_length, embedding_dim].

    Returns:
    - tf.Tensor: A 2D tensor with shape [batch_size, embedding_dim], containing the max-pooled values.
    """
    # Max pooling across the sequence length dimension (axis=1)
    return tf.reduce_max(input_tensor, axis=1)

@tf.function
def classify_sequences(inputs, classification_weights):
    inputs = max_pooling(inputs)


    # Linear transformation
    output = tf.matmul(inputs, classification_weights[0]["class_weights"])



    if BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY:

        # Apply sigmoid activation for multi-label classification
        probabilities = tf.nn.sigmoid(output)

        # Adjust thresholds for rare classes as needed
        thresholds = tf.constant([SPAM_THRESHOLD, TOXIC_THRESHOLD, OBSCENE_THRESHOLD, THREAT_THRESHOLD, INSULT_THRESHOLD, IDENTITY_HATE_THRESHOLD, NEUTRAL_THRESHOLD], dtype=tf.float32)

        # Apply threshold to get binary labels
        labels = tf.cast(probabilities > thresholds, tf.int32)

        # Implemented in conjunction with the loss function instead
        # Enforce the neutral condition: if neutral (last label) is 1, all others must be 0
        neutral_mask = labels[:, -1]  # Extract the neutral column (batch_size,)

        # Set all other labels to 0 where neutral is 1
        output = tf.where(tf.expand_dims(neutral_mask, axis=-1) == 1,
                          tf.concat([tf.zeros_like(labels[:, :-1]), tf.expand_dims(neutral_mask, axis=-1)], axis=-1),
                          labels)

    return output