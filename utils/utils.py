import json
import pandas as pd
from tqdm import tqdm
import random
import os
import numpy as np
import torch

# =========================================================================================
# Seed everything for deterministic results
# =========================================================================================
def seed_everything(seed):
    '''设置随机种子'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_topic_tree(input_topic_df):
    '''
    生成主题树
    Args:
        input_topic_df: topic_df
    Returns:
        df: 包含主题树的 DataFrame, 其中包含列: "id", "topic_tree", "reverse_level"
    '''
    df = pd.DataFrame()

    # 遍历所有唯一的channel
    for channel in tqdm(input_topic_df['channel'].unique()):
        # 筛选出当前 channel 的数据，并重置索引
        channel_df = input_topic_df[(input_topic_df['channel'] == channel)].reset_index(drop=True)
        # 对当前 channel 的 level 进行排序
        for level in sorted(channel_df.level.unique()):
            # 对于 level 为 0 的情况，先为该主题创建一个 topic tree 列，该列为该主题的 title
            if level == 0:
                topic_tree = channel_df[channel_df['level'] == level]['title'].astype(str)
                topic_tree_df = pd.DataFrame([channel_df[channel_df['level'] == level][['id']], topic_tree.values]).T
                # id -> child_id, topic_tree -> topic_tree
                topic_tree_df.columns = ['child_id', 'topic_tree']
                # 将主题树与原始数据合并
                channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(['child_id'], axis=1)

            # 创建 topic tree 列后，将父节点和子节点合并，其中 parent_id = child_id
            topic_df_parent = channel_df[channel_df['level'] == level][['id', 'title', 'parent', 'topic_tree']]
            topic_df_parent.columns = 'parent_' + topic_df_parent.columns

            topic_df_child = channel_df[channel_df['level'] == level + 1][['id', 'title', 'parent', 'topic_tree']]
            topic_df_child.columns = 'child_' + topic_df_child.columns

            # 合并父子节点，生成 topic_df_merged
            topic_df_merged = topic_df_parent.merge(topic_df_child, left_on='parent_id', right_on='child_parent')[
                ['child_id', 'parent_id', 'parent_title', 'child_title', 'parent_topic_tree']]

            # topic tree 是父 topic tree 加上当前子级的 title: 'parent > child'
            topic_tree = topic_df_merged['parent_topic_tree'].astype(str) + ' > ' + topic_df_merged['child_title'].astype(str)

            # 创建包含子节点 id 和 topic_tree 的 DataFrame
            topic_tree_df = pd.DataFrame([topic_df_merged['child_id'].values, topic_tree.values]).T
            topic_tree_df.columns = ['child_id', 'topic_tree']

            # 将 topic_tree_df 与 channel_df 合并，并删除 'child_id' 列
            channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(['child_id'], axis=1)
            
            # 如果存在 'topic_tree_y' 列，则将其与 'topic_tree_x' 列合并
            if 'topic_tree_y' in list(channel_df.columns):
                # combine_first: 用非空值填充空值, 如果两个Series 都不为空,优先使用前者。
                channel_df['topic_tree'] = channel_df['topic_tree_x'].combine_first(channel_df['topic_tree_y'])
                channel_df = channel_df.drop(['topic_tree_x', 'topic_tree_y'], axis=1)
        
        # 将当前 channel 的数据添加到总的 DataFrame 中
        df = pd.concat([df, channel_df], ignore_index=True)

    # 计算每个 channel 的最大 level，并添加到 DataFrame
    df = df.merge(df.groupby("channel")["level"].max().rename("max_level_of_channel").reset_index(),
                  how="left",
                  on="channel")
    # 逆序 level
    df["reverse_level"] = df["max_level_of_channel"] - df["level"]

    return df[["id", "topic_tree", "reverse_level"]]



def read_config():
    '''读取config文件'''
    f = open('config.json')
    config = json.load(f)
    # config["supervised_model"]["betas"] = eval(config["supervised_model"]["betas"])
    return config