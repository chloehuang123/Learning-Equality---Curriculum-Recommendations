[Kaggle: ]()

In this competition, we need to provide high-quality course recommendations for a learning platform. We accomplish this by using a multi-stage process, first dividing the data into training/validation sets and performing text processing. For different languages and content types, we create special tokens and use these tokens to build the model's input. We choose a Sentence-Transformer model with multilingual capabilities, such as AIDA-UPM/mstsb-paraphrase-multilingual-mpnet-base-v2, for Stage 1 training. After completing the training, we use the kNN method of the RAPIDS library to retrieve the top 100 most similar content embeddings for each topic based on Stage 1 embeddings. Then, we train a Stage 2 Cross-Encoder model, using the Stage 1 trained model as a basis. Finally, we adjust the classification threshold based on the F2-score obtained on the Grouped Stratified K-Fold validation set and fill in the highest-scoring Content ID for Topic IDs with no recommended content.

Throughout this process, we attempted to adjust parameters such as Epochs, Batch Size, and learning rate. Through these optimizations and improvements, we ultimately achieved a 0.xxx Top x% score in the competition.
