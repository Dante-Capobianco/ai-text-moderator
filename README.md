# ai-text-moderator
AI model flagging potentially harmful text-based content, such as violent, explicit, or sexual content, hate speech, insults, or spam

### Download Dataset (as alternative to downloading directly from data/raw & data/clean)
[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1oZYg86yW4SJTQVdV3iR5N0IkcwxV_0zP?usp=sharing)


Steps to setup
1. Clone repo to local IDE
2. Install Python plugin (i.e. on marketplace on IntelliJ)
3. Select Python as SDK
4. Install neccessary dependencies (found in setup.py)

make note of performing gpu optimziations using tensorflow-gpu

consider adding weight in embeddings for training only if rare labels involved in certain sequences and/or greater weight in loss finction for mistake on rare classes

### Citations

- **Cormack, G. V. (2007). TREC 2007 Spam Track Public Corpus [Data set].** University of Waterloo. Available at: [https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html)
- **Kaggle. (2019). Jigsaw Unintended Bias in Toxicity Classification [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
- **Kaggle. (2018). Jigsaw Toxic Comment Classification Challenge [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)