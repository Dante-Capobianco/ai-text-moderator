#Tokenize text input (given an input set of data based on training/trainer.py)
#Put results of different methods (different ways of handling padding, truncation, long sequences) & distribution of token lenghts in Jupyter notebook & adjust maximum sequence lengths
import os
import pickle

from socialMediaClassificationTransformer.training.config import MAX_TOKEN_SIZE
import tensorflow as tf


# Function to load vocab list
def load_vocab(file_path):
    vocab={}

    with open(file_path, 'r', encoding='utf-8') as file:
        for index, token in enumerate(file):
            token = token.strip()
            vocab[token] = index
    return vocab
# Function to load data from a .pkl file with error handling
def load_data(file_path):
    with open(file_path, 'rb') as file:
        try:
            return pickle.load(file)
        except EOFError as e:
            raise EOFError(f"Error loading {file_path}: {e}")
#Tokenize the cleaned data, identifying the largest part of each word matching an entry in vocab
def wordpiece_tokenize(text, vocab):
    tokens = []
    for word in text.split():
        start = 0
        while start < len(word):
            end = len(word)
            current_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr  # Add WordPiece prefix for subwords
                if substr in vocab:
                    current_substr = substr
                    break
                end -= 1
            if current_substr is None:
                tokens.append('[UNK]')
                break
            tokens.append(current_substr)
            start = end
    return tokens

def convert_tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab['[UNK]']) for token in tokens]
#Convert cleaned data to tokens, then convert to numerical ID
def tokenize_dataset(data, vocab):
    tokenized_data = []
    for entry in data:
        original_text = entry['text']
        tokens = wordpiece_tokenize(original_text, vocab)
        # Add the CLS token at the beginning
        tokens = ['[CLS]'] + tokens
        token_ids = convert_tokens_to_ids(tokens, vocab)

        # Truncate sequences longer than MAX_TOKEN_SIZE
        if len(token_ids) > MAX_TOKEN_SIZE:
            token_ids = token_ids[:MAX_TOKEN_SIZE]
        # Pad sequences shorter than MAX_TOKEN_SIZE
        elif len(token_ids) < MAX_TOKEN_SIZE:
            padding_length = MAX_TOKEN_SIZE - len(token_ids)
            token_ids.extend([vocab['[PAD]']] * padding_length)

        # Save tokenized entry
        tokenized_entry = {
            "token_ids": token_ids,
            "attention_mask": [0 if token_id == 0 else 1 for token_id in token_ids],
            "correct_labels": list(entry['labels'].values())
        }
        tokenized_data.append(tokenized_entry)

    return tokenized_data

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def serialize_example(token_ids, attention_mask, correct_labels):
    """Creates a tf.train.Example message ready to be written to a file."""
    feature = {
        'token_ids': _bytes_feature(token_ids),
        'attention_mask': _bytes_feature(attention_mask),
        'correct_labels': _bytes_feature(correct_labels),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

#Write each entry to a TfRecord
def save_tokenized_data(file_path, tokenized_data):
    with tf.io.TFRecordWriter(file_path) as writer:
        for item in tokenized_data:
            token_ids = item['token_ids']
            attention_mask = item['attention_mask']
            correct_labels = item['correct_labels']
            example = serialize_example(token_ids, attention_mask, correct_labels)
            writer.write(example)

if __name__ == "__main__":
    #Use GPU & set up memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    vocab = load_vocab('../data/vocab/vocab.txt')
    # Load and inspect each file, then tokenize
    try:
        train = load_data("../data/cleaned/cleaned_train.pkl")
        tokenized_train = tokenize_dataset(train, vocab)
        test = load_data("../data/cleaned/cleaned_test.pkl")
        tokenized_test = tokenize_dataset(test, vocab)
        val = load_data("../data/cleaned/cleaned_validation.pkl")
        tokenized_validation = tokenize_dataset(val, vocab)

        for data in [tokenized_train,
                         tokenized_test,
                           tokenized_validation]:
            for entry in data[:5]:
                print(entry)

            # Paths to save the tokenized datasets
        output_dir = '../data/tokenized/'
        # Save tokenized data to TFRecord files
        save_tokenized_data(os.path.join(output_dir, 'tokenized_train.tfrecord'), tokenized_train)
        save_tokenized_data(os.path.join(output_dir, 'tokenized_test.tfrecord'), tokenized_test)
        save_tokenized_data(os.path.join(output_dir, 'tokenized_validation.tfrecord'), tokenized_validation)



    except Exception as e:
        print(f"Failed to process: {e}")