#Load raw data, initial preprocessing (ie filtering missing/unncessary data, lowercasing, remove special char), visualize distribution of labels & results of preprocessing techniques in Jupyter notebook, split into training, validation, and testing (with balance between all labels)
#Store processed data in pkl
#Note: Jigsaw unintentional bias test.csv does not include labels

import csv
import re
import uuid
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import pickle

# Display occurrence of each label & average characters per dataset
def analyze_data(data):
    label_counts = Counter()
    char_counts = []

    for entry in data:
        labels = entry['labels']
        char_counts.append(len(entry['text']))

        for label, value in labels.items():
            if value == 1:
                label_counts[label] += 1

    avg_char_count = np.mean(char_counts)
    return label_counts, avg_char_count

def generate_random_id():
    # Generate a short random ID (taking the first 8 characters of a UUID) so every message has an ID
    return str(uuid.uuid4())[:8]

#Apply various cleaning strategies before tokenizing
def clean_text(message):
    pattern = r"(Content-Type:|Subject:|Re:|User-Agent:|MIME-Version:|Content-Transfer-Encoding:|name=|charset=|boundary=|Content-ID:|format=|Content-Disposition:|filename=|Message-ID:|X-Accept-Language:|To:|References:|In-Reply-To:|From:|Date:)\s?\S*"

    message = re.sub(pattern, "", message)

    message = message.replace('"""', '"').replace('!','.').replace('-', ' ')
    message = message.lower()
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    message = re.sub(email_pattern, '[EMAIL]', message)
    url_pattern = r'\b(?:https?|ftp):\/\/[^\s/$.?#].[^\s]*|\bwww\.[^\s/$.?#].[^\s]*|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:[^\s]*)?'
    message = re.sub(url_pattern, '[URL]', message)
    message = re.sub(r'[^a-zA-Z0-9.,\s]', '', message)
    # Replace multiple white spaces with a single space
    message = re.sub(r'\s+', ' ', message)

    return message.strip()

#Open each CSV file, clean it, then create the data entry with an id, the message, and the scoring across the labels
def process_data(input_file):
    data = []
    csv.field_size_limit(1000000)
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            input = row['comment_text'] if 'comment_text' in row else row['message']
            if (input is not None and input != "" and len(input) < 15000):
                message = clean_text(input)
                if (message is not None and message != "" and len(message) > 0):

                    labels = {
                    "spam": int(row['label']) if 'label' in row else 0,
                    "toxic": 1 if (('severe_toxic' in row and float(row['severe_toxic']) > 0.5) or ('toxic' in row and float(row['toxic']) > 0.5)) else 0,
                        "obscene": 1 if (('obscene' in row and float(row['obscene']) > 0.5) or ('sexual_attack' in row and float(row['sexual_attack']) > 0.5)) else 0,
                        "threat": 1 if ('threat' in row and float(row['threat']) > 0.5) else 0,
                        "insult": 1 if ('insult' in row and float(row['insult']) > 0.5) else 0,
                        "identity_hate": 1 if (('identity_hate' in row and float(row['identity_hate']) > 0.5) or ('identity_attack' in row and float(row['identity_attack']) > 0.5)) else 0,
                        "neutral": 1 if all((label not in row or float(row[label]) <= 0.5) for label in ['toxic', 'severe_toxic', 'obscene', 'sexual_attack', 'threat', 'insult', 'identity_hate', 'identity_attack', 'label']) else 0
                    }


                    data_entry = {
                        "id": row['id'] if 'id' in row else generate_random_id(),
                        "text": message,
                        "labels": labels
                    }
                    data.append(data_entry)
    return data



#Split the data across 3 sets with an 80-10-10 split for training, testing, and validation
#Rare classes occurring 1-2 times are avoided to use stratified splitting
def stratified_split(data, split_ratios):
    labels_combined = []
    rare_class_label = (0, 0, 1, 1, 0, 1, 0)
    for entryItem in data[:]:
        # Create a tuple of labels as a combined label for stratification
        label_tuple = tuple(entryItem['labels'].values())
        if label_tuple != rare_class_label:
            labels_combined.append(label_tuple)
        else:
            data.remove(entryItem)


    split_indices = StratifiedShuffleSplit(n_splits=1, test_size=split_ratios[1] + split_ratios[2], random_state=42)

    rare_class_labels = [
        (0, 1, 0, 1, 1, 1, 0),
        (0, 0, 1, 1, 1, 1, 0)
    ]
    train_idx, temp_idx = next(split_indices.split(np.zeros(len(data)), labels_combined))
    temp_data = [data[i] for i in temp_idx if labels_combined[i] not in rare_class_labels]
    split_temp = StratifiedShuffleSplit(n_splits=1, test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]), random_state=42)

    # Filter out the rare class labels from temp_idx
    filtered_temp_labels = [labels_combined[i] for i in temp_idx if labels_combined[i] not in rare_class_labels]

    label_counts = Counter(map(tuple, filtered_temp_labels))
    print(label_counts)

    val_idx, test_idx = next(split_temp.split(np.zeros(len(temp_data)), filtered_temp_labels))

    train_data = [data[i] for i in train_idx]
    val_data = [temp_data[i] for i in val_idx]
    test_data = [temp_data[i] for i in test_idx]

    return train_data, val_data, test_data

#Check if empty messages are found
def check_empty_or_null(data):
    null_or_empty_count = sum(1 for entry in data if entry['text'] is None or entry['text'].strip() == "")
    return null_or_empty_count

# Function to save data as a .pkl file
def save_as_pkl(data, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    all_data = []
    for file in ['raw/train.csv', 'raw/train_jigsaw_bias.csv', 'raw/test_public_expanded_jigsaw_bias.csv', 'raw/test_private_expanded_jigsaw_bias.csv','raw/train_spam.csv']:
        cleaned_data = process_data(file)
    all_data.extend(cleaned_data)
    # Print the first few cleaned data entries to verify
    for entry in cleaned_data[:5]:
        print(entry)

    train_data, val_data, test_data = stratified_split(all_data, [0.8, 0.1, 0.1])


    train_distribution, train_avg_chars = analyze_data(train_data)
    val_distribution, val_avg_chars = analyze_data(val_data)
    test_distribution, test_avg_chars = analyze_data(test_data)

    print(f"Training set: {train_distribution}, Avg chars: {train_avg_chars}")
    print(f"Validation set: {val_distribution}, Avg chars: {val_avg_chars}")
    print(f"Test set: {test_distribution}, Avg chars: {test_avg_chars}")

    # Check for null or empty text fields in each dataset
    train_null_or_empty = check_empty_or_null(train_data)
    val_null_or_empty = check_empty_or_null(val_data)
    test_null_or_empty = check_empty_or_null(test_data)

    print(f"Train data has {train_null_or_empty} entries with null or empty text fields.")
    print(f"Validation data has {val_null_or_empty} entries with null or empty text fields.")
    print(f"Test data has {test_null_or_empty} entries with null or empty text fields.")

    # Define the existing output directory
    output_dir = 'cleaned/'

    # Save the train, validation, and test data as .pkl files
    save_as_pkl(train_data, 'cleaned_train.pkl')
    save_as_pkl(val_data, 'cleaned_validation.pkl')
    save_as_pkl(test_data, 'cleaned_test.pkl')

    print("Data saved successfully in the 'cleaned/' folder.")

