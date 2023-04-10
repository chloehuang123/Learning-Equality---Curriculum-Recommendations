import pandas as pd
from tqdm import tqdm
import gc

from sentence_transformers import SentenceTransformer, models, InputExample, losses

from cuml.neighbors import NearestNeighbors
import cupy as cp
import torch

from .utils import generate_topic_tree


def get_neighbors(topic_df,
                  content_df,
                  config_obj):
    # 创建无监督模型以提取 embeddings
    model = SentenceTransformer(config_obj["unsupervised_model"]["save_name"])
    model = model.to("cuda")

    # topics embeddings
    topics_preds = model.encode(topic_df["model_input"], show_progress_bar=True, convert_to_tensor=True)
    topics_preds_gpu = cp.asarray(topics_preds) # 将数据转换为GPU

    # content embeddings
    content_preds = model.encode(content_df["model_input"], show_progress_bar=True, convert_to_tensor=True, batch_size=100)
    content_preds_gpu = cp.asarray(content_preds) # 将数据转换为GPU

    # Release memory
    torch.cuda.empty_cache()
    gc.collect()

    # KNN model
    print(' ')
    print('Training KNN model...')
    # 定义KNN模型
    neighbors_model = NearestNeighbors(n_neighbors=config_obj["unsupervised_model"]["top_n"], metric='cosine')
    # 训练KNN模型
    neighbors_model.fit(content_preds_gpu)
    # 推理：获取最近的top_n个邻居
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance=False)

    # 组成预测结果
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content_df.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    topic_df['predictions'] = predictions

    # Release memory
    del topics_preds, content_preds, topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    return topic_df, content_df


def build_training_set(topic_df, content_df, mode="local"):
    '''构建监督训练的训练数据集'''

    # 创建用于训练的列表
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []

    # 遍历每个 topic
    for k in tqdm(range(len(topic_df))):
        row = topic_df.iloc[k]
        topics_id = row['id']
        topics_title = row['model_input']
        predictions = row['predictions'].split(' ')

        # 如果模式为本地，则获取 ground truth
        if mode == "local":
            ground_truth = row['content_ids'].split(' ')

        for pred in predictions:
            content_title = content_df.loc[pred, 'model_input']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)

            # 如果pred在ground truth中，target为1，否则为0
            if mode == "local":
                if pred in ground_truth:
                    targets.append(1)
                else:
                    targets.append(0)

    # DataFrame
    train = pd.DataFrame(
        {'topics_ids': topics_ids,
         'content_ids': content_ids,
         'model_input1': title1,
         'model_input2': title2
         }
    )
    if mode == "local":
        train["target"] = targets

    return train


def read_data(data_path,
              config_obj,
              read_mode="all"):
    topics = pd.read_csv(data_path + 'topics.csv')
    content = pd.read_csv(data_path + 'content.csv')

    if read_mode != "all":
        correlations = pd.read_csv(data_path + 'correlations.csv')
    else:
        correlations = None
    topic_trees = generate_topic_tree(topics) # Generating a topic tree DataFrame

    if read_mode != "all":
        splits = pd.read_csv("train_test_splits.csv")  # Reading the split information of the training and test data
        topics = topics[topics.id.isin(splits[splits.fold == read_mode].id)].reset_index(drop=True) # Filtering the topics of the train/valid dataset based on the read_mode

    topics = topics.merge(topic_trees, how="left", on="id") # Merge topic data with a topic tree
    del topic_trees # Delete the topic tree variable to free up memory
    gc.collect()

    topic_tokens = generate_topic_model_input(input_df=topics) # To generate input data for the topic model
    content_tokens = generate_content_model_input(input_df=content) # To generate input data for the content model

    unq_tokens = set(topic_tokens + content_tokens + ["nan"]) # the unique tokens

    # Sort by title length to speed up the process
    topics['length'] = topics['title'].apply(lambda x: len(x)) # topic title length
    content['length'] = content['title'].apply(lambda x: len(x)) # content title length
    topics.sort_values('length', inplace=True) # Sort topics by title length
    content.sort_values('length', inplace=True) # Sort content by title length

    # Delete lengthcols
    topics.drop(['length'], axis=1, inplace=True)
    content.drop(['length'], axis=1, inplace=True)

    # reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    if read_mode != "all":
        print(f"correlations.shape: {correlations.shape}")

    return topics, content, correlations, unq_tokens


def generate_token_features(input_df, token_features):
    """
    Args:
        input_df
        token_features: 需要生成token的列名list
    Returns:
        Tuple of (带有额外模型输入列的df, Unique Special Tokens)
        Tuple of (Dataframe with additional model input column, Unique Special Tokens)
    """
    token_feature_set = None # 初始化token特征集
    special_tokens = [] # 初始化 sep token 列表

    # 遍历 token_features 中的每个特征列
    for token_feature in token_features:
        # 为当前特征生成 sep token 字符串
        token_feature_str = "[<[" + token_feature + "_" + input_df[token_feature].astype(str) + "]>]"
        special_tokens += set(token_feature_str.values)  # 将生成的 sep token 添加到列表中

        # 将生成的 sep token 字符串添加到 token_feature_set 中
        if not isinstance(token_feature_set, pd.Series):
            token_feature_set = token_feature_str
        else:
            token_feature_set += " " + token_feature_str

    # 返回 token_feature_set 和 Unique Special Tokens
    return token_feature_set, special_tokens


def generate_topic_model_input(input_df):
    """
    Args:
        input_df
    Returns:
        修改 input_df
        Dataframe with additional model input column
    """
    input_df.fillna("nan", inplace=True) # 用"nan"填充空值

    # token特征列表
    token_features = ["language",]

    # 生成 token_feature_text 和 Special Tokens
    token_feature_text, special_tokens = generate_token_features(input_df=input_df, token_features=token_features)

    # 生成 model_input，将 token_feature_text、title、topic_tree 和 description 组合起来，并将所有文本转换为小写
    input_df["model_input"] = (
            token_feature_text +
            " [<[topic_title]>] " + input_df["title"].astype(str) +
            " [<[topic_tree]>] " + input_df["topic_tree"].astype(str) +
            " [<[topic_desc]>] " + input_df["description"].astype(str)
    ).str.lower()

    del token_feature_text

    input_df.drop(['description', 'channel', 'category',
                   'level', 'parent', 'has_content'],
                  axis=1,
                  inplace=True)
    gc.collect()

    # return special_tokens, cause input_df has been changed
    return special_tokens


def generate_content_model_input(input_df):
    """
    Args:
        input_df
    Returns:
        修改 input_df
        Dataframe with additional model input column
    """

    input_df.fillna("nan", inplace=True) # 用"nan"填充空值

    # token特征列表
    token_features = ["language", "kind"]

    # 生成 token_feature_text 和 Special Tokens
    token_feature_text, special_tokens = generate_token_features(input_df=input_df, token_features=token_features)

    # 生成 model_input，将 token_feature_text、title、description 和 text 组合起来，限制长度为512个单词，并将所有文本转换为小写
    input_df["model_input"] = (
            token_feature_text +
            " [<[cntnt_title]>] " + input_df["title"].astype(str) +
            " [<[cntnt_desc]>] " + input_df["description"].astype(str) +
            " [<[cntnt_text]>] " + input_df["text"].astype(str)
    ).apply(lambda x: " ".join(x.split()[:512])).str.lower()

    del token_feature_text

    input_df.drop(['description', 'kind', 'text', 'copyright_holder', 'license'],
                  axis=1,
                  inplace=True)
    gc.collect()

    # 返回 special_tokens, input_df 在函数内部被修改
    return special_tokens