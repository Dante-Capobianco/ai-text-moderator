# Social Media Multi-Label Text Classification Transformer
As CapoDev’s founder, I am planning on pursuing a reimagined social platform down the line in our app development timeline. Content moderation is a cornerstone for online security across the Internet, yet it can be challenging for an environment with both youth and adults to keep youth safe from cyberbullies. In fact, over 60% of youth are victims of harmful online interactions. My goal is to offer a large-scale solution to strict moderation with special focus on supporting youth.

Through a transformer encoder architecture with innovative features, I have developed a Python model that identifies if short/long-form text is spam, toxic, threat, insult, identity hate, obscene, and/or neutral. With ~3.4 million data points from 3 credible datasets and ~97.6% accuracy on test/validation data, this offers a revolutionary, speedy way to automate content review. 

Some of the most inventive features include:
- NVIDIA GPU optimization using Tensorflow-GPU-10.0.0, CUDA-11.2, cuDNN-8.1, TFRecords for tokenized data, and initializing Tensorflow graph execution, parallel batch loading/processing & dynamic memory growth
- One-of-a-kind parameters for customizable hypertuning - i.e. enabling either binary cross entropy applied to binary probabilities (with thresholds & a neutral exclusivity rule applied) or sigmoid cross entropy on logits for loss calculation
- Considering a combination of validation loss value decreases, validation accuracy improvements & training-to-validation loss gaps (to avoid overfitting) to determine if each epoch’s weights should be checkpointed; plus, a patience parameter is used to early stop training if 1+ epochs show no improvement

With a proven performance & able to review hundreds of inputs within seconds, this transformer can improve the cleanliness of online interactions.


## Steps to Configure & Run

### 1. Download Dataset
[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1oZYg86yW4SJTQVdV3iR5N0IkcwxV_0zP?usp=sharing)
*Note: if the raw_train_test_val_data.zip is hidden, you can either jump straight to downloading the cleaned/tokenized data, or download the appropriate datasets from the citation links below (ensuring the naming convention matches the naming in **Step 4.3**)

### 2. Prerequisites
Ensure you have the prerequisites: Git, Python 3.10.0 as your SDK, CUDA 11.2, cuDNN 8.1 (for IntelliJ users, ensure you have the Python Community Edition plugin as your **only active Python plugin**.

### 3. Download Source Code
1 method is to:
1. Follow the instructions [here](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop) to clone to your local computer
2. Open your IDE & in the terminal enter this:
   ```bash
   pip install -r requirements.txt
A second method is shown here:
1. Ensure you have the prerequisites: Git, Python 3.10.0, CUDA 11.2, cuDNN 8.1
2. ```bash
   git clone https://github.com/Dante-Capobianco/ai-text-moderator.git
   cd ai-text-moderator
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

### 4. Running the Program
*Skip the first 2 steps if you have downloaded the tokenized data (skip the first step if you have downloaded the cleaned data)
1. Run the data_loader.py to generate cleaned datasets in the data/cleaned directory
2. Run the tokenizer.py to generate tokenized datasets in the data/tokenized directory (if you wish to adjust the MAX_TOKEN_SIZE parameter in the config.py file, it must be done in this step - the tokenized data will have to be regenerated for any future changes to this parameter)
3. By now, your data directory structure, with the original datasets downloaded, should look like this: ![Screenshot 2024-08-24 162039](https://github.com/user-attachments/assets/10baafad-524c-4b0e-8ef9-307ca9c4192f)
4. Run the train.py file - this is where you may need to adjust parameters in the config.py file to avoid OOM errors & optimize the model
5. Review the results from training and validation after each epoch to determine which set of checkpointed weights are performing most effectively (note that the number of saved weight sets can be adjusted through the max_to_keep setting when initializing the checkpoint object in the train.py, real_world_data_predict.py, and predict.py)
6. Once satisfied with a set of weights, run the saved model in predict.py to test it on the test dataset. A unique feature of this model (elaborated on in the report) is that parameters can be tuned during testing (such as thresholds to adjust the labelling sensitivity under each category)
7. Once satisfied with the saved model & tuned test parameters, it can be further tested & tuned using live user input in the real_world_data_predict.py file


## Report
Available at: [CapoDev Multi-Label Text Classification Transformer](https://docs.google.com/document/d/1tZsWFJ38rv13aUw0n1uIgANljv0hqC0YOihRFfJ73fg/edit?usp=sharing)

## CapoDev
Learn more about **CapoDev** on **[Instagram](https://www.instagram.com/capodevapps/), [TikTok](https://www.tiktok.com/@capodevapps?is_from_webapp=1&sender_device=pc), [LinkedIn](https://www.linkedin.com/in/dante-capobianco/), [Facebook](https://www.facebook.com/share/yqn765FgofCrR2jr/?mibextid=LQQJ4d), and [YouTube](https://youtube.com/@capodevapps?si=DF53TrDJOSbCm4Hj)**. Dante, the visionary behind CapoDev, leads the mobile & web app development processes with the mission of developing applications that impact the daily routines of the everyday consumer. Currently, Dante is leading the development of a revolutionary social app addressing major issues currently faced when participating on social media platforms. 

## Citations

- **Cormack, G. V. (2007). TREC 2007 Spam Track Public Corpus [Data set].** University of Waterloo. Available at: [https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html)
- **Kaggle. (2019). Jigsaw Unintended Bias in Toxicity Classification [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
- **Kaggle. (2018). Jigsaw Toxic Comment Classification Challenge [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
