#Store hyperparameters (learning rate, batch size, epochs, number of layers)

"""

"""

DATA_PATH = "data/datasets/"
TOKENIZER_PATH = "data/tokenizer.py"
SAVED_MODEL_PATH = "trained_models/"
#Number of text inputs
BATCH_SIZE=32
DIMENSION=128
#Need to redo tokenization of data if change token size
MAX_TOKEN_SIZE=500
LEARNING_RATE=0.0001
DECAY_RATE=0.96
ADAMW_WEIGHT_DECAY=0.0001
#Note: Approximately 6900 steps/batches at batch_size = 256; 8x for batch_size = 32
WARMUP_STEPS=100
DECAY_STEPS=1400 #Roughly 14x warmup_steps size
TOTAL_VALIDATION_BATCHES=200
TOTAL_TESTING_BATCHES=500
VOCAB_SIZE=30524
EPOCHS=10
#Number of epochs for there to be no loss decrease or accuracy increase before stopping training
PATIENCE=2
#Reasonable gap between 0.01 and 0.1
ACCEPTABLE_LOSS_GAP=0.1
ATTENTION_HEADS=4
DROPOUT_RATE=0.1
NUM_LAYERS=3
PENALTY_FACTOR=1
INCLUDE_CLS=True
INCLUDE_DYNAMIC_BALANCING=False
MASK_IN_EMBEDDINGS=True
BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY=False
#Thresholds for if a message should be considered under each label
"""
other options include:

proportional to frequency
thresholds = {
    'spam': 0.019747893987345476,
    'toxic': 0.007034362122579018,
    'obscene': 0.00778904515831817,
    'threat': 0.0016358941064848926,
    'insult': 0.04432700556884076,
    'identity_hate': 0.004507948831819861,
    'neutral': 0.9489578502246118
}

log scaling
"""
SPAM_THRESHOLD=0.5
TOXIC_THRESHOLD=0.3
OBSCENE_THRESHOLD=0.3
THREAT_THRESHOLD=0.2
INSULT_THRESHOLD=0.3
IDENTITY_HATE_THRESHOLD=0.5
NEUTRAL_THRESHOLD=0.9