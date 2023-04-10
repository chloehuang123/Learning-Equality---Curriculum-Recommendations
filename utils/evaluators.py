from typing import List, Tuple, Dict, Set, Callable, Optional
from sentence_transformers.util import cos_sim
from torch import Tensor
import heapq
from sentence_transformers import evaluation
import os
from tqdm import trange
import torch
import numpy as np

class InformationRetrievalEvaluator(evaluation.SentenceEvaluator):
    """
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    给定一组查询和一个大型语料库集。它将为每个查询检索最相似的前k个文档。
    并测量平均逆向排名(MRR), Recall@k和 NDCG
    """

    def __init__(self,
                 queries: Dict[str, str], # qid => query，查询字典，qid表示查询编号，query表示查询内容
                 corpus: Dict[str, str],  # cid => doc，  语料字典，cid表示语料编号，doc表示语料内容
                 relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]，有用文档字典，qid表示查询编号，Set[cid]为与该查询相关的有用文档id集合
                 corpus_chunk_size: int = 50000, # 语料分块大小，默认为50000
                 mrr_at_k: List[int] = [10], # MRR@k，k取值列表，默认为[10]
                 ndcg_at_k: List[int] = [10], # NDCG@k，k取值列表，默认为[10]
                 accuracy_at_k: List[int] = [1, 3, 5, 10], # 准确率@k，k取值列表，默认为[1, 3, 5, 10]
                 precision_recall_at_k: List[int] = [1, 3, 5, 10], # 精确率@k，k取值列表，默认为[1, 3, 5, 10]
                 map_at_k: List[int] = [100], # MAP@k，k取值列表，默认为[100]
                 show_progress_bar: bool = False, # 是否显示进度条，默认为False
                 batch_size: int = 32, # batch_size，默认为32
                 name: str = '', # 评估器名称，默认为空
                 write_csv: bool = True, # 是否将结果写入csv文件，默认为True
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim}, # 得分函数，得分越高说明相似度越高
                 main_score_function: str = None # 主要得分函数，默认为None
                 ):
        
        # 获取有用文档的queries_ids
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        # 查询文本列表
        self.queries = [queries[qid] for qid in self.queries_ids]
        # 语料文本列表
        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.precision_recall_at_k = precision_recall_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) # 得分函数名称列表
        self.main_score_function = main_score_function
        
        # 设置评估器名称
        if name:
            name = "_" + name
        
        # 设置csv文件名和表头信息
        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        """
        计算模型在给定数据上的评估指标, 并将结果保存到csv文件中
        """
        # 根据epoch和steps设置输出信息
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        
        # 计算指标
        scores = self.compute_metrices(model, *args, **kwargs)

        # 将结果写入csv文件
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()
            
        # 返回主要指标得分
        if self.main_score_function is None:
            return max([scores[name]['recall@k'][max(self.precision_recall_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['recall@k'][max(self.precision_recall_at_k)]

    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(self.precision_recall_at_k)

        # 计算查询的embedding
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # 遍历语料库的chunk
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #语料库的 chunk 进行 embedding
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # 计算余弦相似度
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # 获取top-k值
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()
                
                # 将 score 添加到查询结果列表 (score, corpus_id)
                # 遍历 query_embeddings 的索引
                for query_itr in range(len(query_embeddings)):
                    # 遍历当前查询结果的 top-k 索引和对应的分数
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        # 计算当前子语料库中的文档在整个语料库中的实际索引
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        # 如果当前查询结果列表的长度小于 max_k
                        if len(queries_result_list[name][query_itr]) < max_k:
                             # 使用 heapq.heappush 将新的分数和对应的语料库 ID 以元组形式添加到查询结果列表中
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id)) # heapq 追踪元组中第一个元素的数量
                        else:
                            # 否则，使用 heapq.heappushpop 将新的分数和对应的语料库 ID 以元组形式添加到查询结果列表中，同时弹出当前最小的分数
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
        
        # 将查询结果列表转换为字典格式 (score, corpus_id) -> {'score': score, 'corpus_id': corpus_id}
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}


        # 计算评分
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # 初始化评分计算值
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}

        # 根据结果计算评分
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # 对查询结果按分数降序排序
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # 计算 Precision 和 Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    # 如果查询结果中的文档ID存在于相关文档集合中，则认为是正确结果，将 num_correct 计数加1
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

        # 计算各个k值下的平均精确率
        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        # 计算各个k值下的平均召回率
        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        # 返回包含各个k值下平均精确率和召回率的字典
        return {'precision@k': precisions_at_k, 'recall@k': recall_at_k}