import numpy as np


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    if model.mode == 'inner':
        pred_y = - model.predict(test)
    else:
        pred_y = model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)