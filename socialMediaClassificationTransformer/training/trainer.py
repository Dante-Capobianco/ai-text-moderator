#Training loop, including batching, loss calculation, backpropagation using the optimizer to update weights, evaluating the model on validation data every 1+ epochs
#Saves model as well

# ../training/trainer.py

from socialMediaClassificationTransformer.models.embedding import init_embedding_weights, init_embedding, backward_pass
from socialMediaClassificationTransformer.models.transformer import run_transformer
from socialMediaClassificationTransformer.models.attention import init_attention_weights
from socialMediaClassificationTransformer.training.config import EPOCHS, DIMENSION


def train(embedding, epochs, embed_weights, attention_weights):
    for epoch in range(epochs):  # Simulate the training epochs
        print(f"Epoch {epoch+1}/{epochs}")
        i = 1

        for batch in embedding:
            print(f"Batch {i}/{len(embedding)}")
            #call transformer to run through layers once
            ouput = run_transformer(batch, embed_weights, attention_weights, True)
            # Assume some loss calculation here (replace with actual implementation)
            # loss = calculate_loss(output)

            # Backpropagation: update weights based on the loss
            # gradient = calculate_gradient(loss) #function in embedding.py
            # embedding_weights = backward_pass(embedding_weights,gradient)
            # gradients = calculate_gradient(loss) #function in attention.py with this format:
            # gradients = {
            #     "W_q": np.random.randn(embedding_dim, embedding_dim),
            #     "W_k": np.random.randn(embedding_dim, embedding_dim),
            #     "W_v": np.random.randn(embedding_dim, embedding_dim),
            #     "W_o": np.random.randn(embedding_dim, embedding_dim),
            # }
            # attention_weights = update_weights(attention_weights,gradients)
            #print(f"Updated Embeddings: {embedding_weights[:5]}")  # Print a preview of updated embeddings
            i += 1

# Example usage
if __name__ == "__main__":

    embedding_weights = init_embedding_weights()
    att_weights = init_attention_weights()
    data = init_embedding("../data/tokenized/tokenized_train.pkl")

    # Start training
    train(data, EPOCHS, embedding_weights, att_weights)
