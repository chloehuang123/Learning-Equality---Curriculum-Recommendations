import numpy as np


def get_pos_score(y_true, y_pred):
    '''Max Recall'''
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split())) 
    # 计算每个元素的交集比率（正确预测数量/真实标签数量）
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

def get_f2_score(y_true, y_pred):
    '''F2 score'''
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    # 真正例数量（真实标签与预测标签的交集数量）
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)]) 
    # 假正例数量（预测标签中不属于真实标签的数量）
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    # 假负例数量（真实标签中未被预测到的数量）
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    # 根据公式计算F2得分
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)