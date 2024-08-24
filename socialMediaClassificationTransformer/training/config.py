#Store parameters (learning rate, batch size, epochs, number of layers)

#Good loss < 0.5; Good F1 > 0.7

DATA_PATH = "data/datasets/"
TOKENIZER_PATH = "data/tokenizer.py"
SAVED_MODEL_PATH = "trained_models/"

#Number of text inputs in each batch
BATCH_SIZE=32
#Embedding dimension for each token
DIMENSION=128
#Need to redo tokenization of data if change token size
MAX_TOKEN_SIZE=500

#Learning schedule changes
LEARNING_RATE=0.0001
DECAY_RATE=0.96
ADAMW_WEIGHT_DECAY=0.0001 #Only applied for AdamW

#Note: Approximately 6900 steps/batches in the training set at batch_size = 256; at batch_size = 32, roughly 55,200
WARMUP_STEPS=100
DECAY_STEPS=1400 #14x warmup_steps size

#Number of batches to validate & test model
TOTAL_VALIDATION_BATCHES=200
TOTAL_TESTING_BATCHES=3000

VOCAB_SIZE=30524

EPOCHS=10

#Number of epochs for there to be no loss decrease or accuracy increase before stopping training
PATIENCE=2

#Reasonable gap between 0.01 and 0.1
ACCEPTABLE_LOSS_GAP=0.1
ACCEPTABLE_ACCURACY_GAP=0.5 #Percentage difference

#Number of splits made in each token's embedding to increase methods of understanding token's context to rest of sequence
ATTENTION_HEADS=8

#For eliminating specific dimensions from consideration throughout various processes during training
DROPOUT_RATE=0.1

#Total layers in the transformer
NUM_LAYERS=6

PENALTY_FACTOR=1 #penalty if neutral is applied with another label

#Whether to include CLS token in classification output
INCLUDE_CLS=True

#To keep proportions in each batch loaded
INCLUDE_DYNAMIC_BALANCING=False

#Apply attention mask in embedding step
MASK_IN_EMBEDDINGS=True

#To apply binary cross entropy instead of sigmoid cross entropy & apply a neutral exclusivity rule
BINARY_CROSS_ENTROPY_WITH_THRESHOLDS_AND_NEUTRAL_EXCLUSIVITY=False

#Thresholds for if a message should be considered under each label
"""
thresholds chosen to be proportional to frequency
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
#THRESHOLDS FOR CONVERTING PROBABILITIES TO PREDICTIONS; MAKE SURE TO WRITE DOWN BEST MODEL'S THRESHOLDS BEFORE EDITING
SPAM_THRESHOLD=0.5
TOXIC_THRESHOLD=0.3
OBSCENE_THRESHOLD=0.3
THREAT_THRESHOLD=0.2
INSULT_THRESHOLD=0.3
IDENTITY_HATE_THRESHOLD=0.5
NEUTRAL_THRESHOLD=0.9
#Best thresholds for training:
# 0.5, 0.3, 0.3, 0.2, 0.3, 0.5, 0.9

#Thresholds for applying to test/real-world data
TEST_SPAM_THRESHOLD=0.5
TEST_TOXIC_THRESHOLD=0.3
TEST_OBSCENE_THRESHOLD=0.3
TEST_THREAT_THRESHOLD=0.2
TEST_INSULT_THRESHOLD=0.3
TEST_IDENTITY_HATE_THRESHOLD=0.5
TEST_NEUTRAL_THRESHOLD=0.9
#Best thresholds for testing:
# 0.5, 0.3, 0.3, 0.2, 0.3, 0.5, 0.9