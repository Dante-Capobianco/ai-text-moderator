#Load raw data, initial preprocessing (ie filtering missing/unncessary data, lowercasing, remove special char), visualize distribution of labels & results of preprocessing techniques in Jupyter notebook, split into training, validation, and testing (with balance between all labels)
#Store processed data in pkl


#balance avg char per comment, number of each label


#PULL THIS INTO LOCAL REPO

import csv
import re

def clean_text(message):
    message = message.replace('"""', '"')
    message = message.lower()
    # Replace multiple white spaces with a single space
    message = re.sub(r'\s+', ' ', message)
    
    return message.strip()
    
#check if message is empty or zero non-whitespace char  to skip then