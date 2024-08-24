# ai-text-moderator
AI model flagging potentially harmful text-based content, such as violent, explicit, or sexual content, hate speech, insults, or spam

### Download Dataset
[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1oZYg86yW4SJTQVdV3iR5N0IkcwxV_0zP?usp=sharing)


Steps to setup
1. Clone repo to local IDE
2. Install Python plugin (i.e. on marketplace on IntelliJ)
3. Select Python as SDK
4. Install neccessary dependencies (found in setup.py)

make note of performing gpu optimziations using tensorflow-gpu

consider adding weight in embeddings for training only if rare labels involved in certain sequences and/or greater weight in loss finction for mistake on rare classes

### Report
Available at: [CapoDev Multi-Label Text Classification Transformer](https://docs.google.com/document/d/1tZsWFJ38rv13aUw0n1uIgANljv0hqC0YOihRFfJ73fg/edit?usp=sharing)

### Presentation
Find **CapoDev Apps** on **[Instagram](https://www.instagram.com/capodevapps/), [TikTok](https://www.tiktok.com/@capodevapps?is_from_webapp=1&sender_device=pc), [Facebook](https://www.facebook.com/share/yqn765FgofCrR2jr/?mibextid=LQQJ4d), and [YouTube](https://youtube.com/@capodevapps?si=DF53TrDJOSbCm4Hj)**. Dante, the visionary behind CapoDev, leads the mobile & web app development processes with the mission of developing applications that impact the daily routines of the everyday consumer. Currently, Dante is leading the development of a revolutionary social app addressing major security & socialization issues present on numerous social platforms. 

### Citations

- **Cormack, G. V. (2007). TREC 2007 Spam Track Public Corpus [Data set].** University of Waterloo. Available at: [https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html)
- **Kaggle. (2019). Jigsaw Unintended Bias in Toxicity Classification [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
- **Kaggle. (2018). Jigsaw Toxic Comment Classification Challenge [Data set].** Available at: [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
