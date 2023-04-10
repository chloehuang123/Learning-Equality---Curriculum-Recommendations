#!/usr/bin/env python
# coding: utf-8
# %%
## 1. Training Unsupervised SentenceTransformer

# %%

import faulthandler
faulthandler.enable()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # set up jupyter env


# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers import datasets


from datasets import Dataset


import warnings
warnings.filterwarnings('ignore')


# %%
# Custom libraries
from utils.unsupervised_utils import read_data
from utils.utils import read_config
from utils.evaluators import InformationRetrievalEvaluator
# %%
os.environ["TOKENIZERS_PARALLELISM"]="true" # Tokens parallelization
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="false" # Turn on advisory warnings


# %%
config = read_config() # Read configuration file


# %%
DATA_PATH = "../raw_data/" # Data path


# %%
# Read training data
topics, content, correlations, _ = read_data(data_path=DATA_PATH, config_obj=config, read_mode="train")

# Add column name prefixes to topic and content, respectively
topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

# groud truth
correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])

# Combining groud truth with topics and content
corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")

# topic_input, content_input -> df
corr["set"] = corr[["topic_model_input", "content_model_input"]].values.tolist()
train_df = pd.DataFrame(corr["set"])

dataset = Dataset.from_pandas(train_df)

train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows # number of training examples

# Create train_examples
for i in range(n_examples):
    example = train_data[i]
    if example[0] == None:
        continue
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))

# %%
# Setting-up the Evaluation
# read verification data
test_topics, test_content, test_correlations, _ = read_data(data_path=DATA_PATH, config_obj=config, read_mode="test")

test_correlations["content_id"] = test_correlations["content_ids"].str.split(" ")
test_correlations = test_correlations[test_correlations.topic_id.isin(test_topics.id)].reset_index(drop=True)
test_correlations["content_id"] = test_correlations["content_id"].apply(set)
test_correlations = test_correlations[["topic_id", "content_id"]] # Keep the topic_id and content_id of the groud truth

# validation gt saved as a dictionary: {topic_id: content_id}
ir_relevant_docs = {
    row['topic_id']: row['content_id'] for i, row in tqdm(test_correlations.iterrows())
}


# %%
# Keep unique topic_id
unq_test_topics = test_correlations.explode("topic_id")[["topic_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
# The only topic_id, merged with the corresponding model_input
unq_test_topics = unq_test_topics.merge(test_topics[["id", "model_input"]], how="left", left_on="topic_id", right_on="id").drop("id", 1)

#Validation training text saved as a dictionary: {topic_id: model_input}
ir_queries = {
    row['topic_id']: row['model_input'] for i, row in tqdm(unq_test_topics.iterrows())
}


# %%
# read all data
all_topics, all_content, _, special_tokens = read_data(data_path=DATA_PATH, config_obj=config, read_mode="all")
# Keep unique content_id
unq_contents = correlations.explode("content_id")[["content_id"]].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
# Unique content_id, combined with the corresponding model_input
unq_contents = unq_contents.merge(all_content[["id", "model_input"]], how="left", left_on="content_id", right_on="id").drop("id", 1)
# Save all the content text as a dictionary: {content_id: model_input}
ir_corpus = {
    row['content_id']: row['model_input'] for i, row in tqdm(unq_contents.iterrows())
}

# %%
evaluator = InformationRetrievalEvaluator(
    ir_queries,  # validation topic_id content
    ir_corpus,   # all content cotent
    ir_relevant_docs, # validation gt
    show_progress_bar=True, # show progress bar
    main_score_function="cos_sim", # main score function
    precision_recall_at_k=[5, 10, 25, 50, 100],  # precision at k
    name='K12-local-test-unsupervised' # evaluator name
)

# %%
# Training
# datasets.NoDuplicatesDataLoader: Duplicate data can be filtered out to ensure that each sentence pair is only fed into the model once
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=config["unsupervised_model"]["batch_size"])


# %%
TARGET_MODEL = config["unsupervised_model"]["base_name"] # pre-trained model
OUT_MODEL = config["unsupervised_model"]["save_name"] # output model


# %%
model = SentenceTransformer(TARGET_MODEL) # Load the pretrained model
model.max_seq_length = config["unsupervised_model"]["seq_len"] # set up maxlen

word_embedding_model = model._first_module() # Get the word vector model

# Add sep token to tokenizer and re-adjust the number of tokens
word_embedding_model.tokenizer.add_tokens(list(special_tokens), special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))


# %%
train_loss = losses.MultipleNegativesRankingLoss(model=model) # Define the loss function

#k% of train data
num_epochs = config["unsupervised_model"]["epochs"] # number of training rounds
warmup_steps = int(len(train_dataloader) * config["unsupervised_model"]["warmup_ratio"]) # Warm-up steps


# %%
model.fit(train_objectives=[(train_dataloader, train_loss)], # Training Data and Loss Function
#           scheduler="constantlr",
#           optimizer_class=Lion,
#           optimizer_params={'lr': 2e-5},
          evaluator=evaluator,  # define evaluator
#           evaluation_steps=400,

          checkpoint_path=f"checkpoints/unsupervised/{OUT_MODEL.split('/')[-1]}", # Path to save checkpoints
          checkpoint_save_steps=len(train_dataloader), # The number of steps to save the checkpoint

          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=OUT_MODEL,
          save_best_model=True,
          use_amp=True # Mixed precision training
          )

