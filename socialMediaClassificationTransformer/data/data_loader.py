#Load raw data, initial preprocessing (ie filtering missing/unncessary data, lowercasing, remove special char), visualize distribution of labels & results of preprocessing techniques in Jupyter notebook, split into training, validation, and testing (with balance between all labels)
#Store processed data in pkl


#balance avg char per comment, number of each label


#PULL THIS INTO LOCAL REPO

import csv
import re

def clean_text(message):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    message = re.sub(email_pattern, '[message]', text)
    message = message.replace('"""', '"').replace('!','.')
    message = message.lower()
    message = re.sub(r'[^a-zA-Z0-9.,\s]', '', message)
    # Replace multiple white spaces with a single space
    message = re.sub(r'\s+', ' ', message)
    
    return message.strip()
    
def process_data(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message = clean_text(row['comment_text'])
            #adjust labels to align with other dataset and labelling goals
            #use jigsaw unintented bias
            #other dataset: one of spams
            labels = {
            "spam": int(row['spam']) if 'spam' in row else 0,
                "toxic": int(row['toxic']),
                "severe_toxic": int(row['severe_toxic']),
                "obscene": int(row['obscene']),
                "threat": int(row['threat']),
                "insult": int(row['insult']),
                "identity_hate": int(row['identity_hate']),
                "neutral": 1 if all(int(row[label]) == 0 for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']) else 0
            }
            data_entry = {
                "text": message,
                "labels": labels
            }
            data.append(data_entry)
    return data

# Example usage
input_file = 'input.csv'  # Replace with your actual input file path
cleaned_data = process_data(input_file)

# Print the first few cleaned data entries to verify
for entry in cleaned_data[:5]
    
#check if message is empty or zero non-whitespace char  to skip then