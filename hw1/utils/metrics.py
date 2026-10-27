import numpy as np


def evaluate(model, history, target, k=10, users=None):
    relevant = target[target.rating >= 4].groupby('u').i.apply(set).to_dict()
    if users is None:
        users = sorted(relevant)
    users = [int(user) for user in users]
    if not users:
        return {'ndcg@10': float('nan'), 'recall@10': float('nan'), 'users': 0}

    seen = history.groupby('u').i.apply(set).to_dict()
    catalog = np.sort(history.i.unique())
    ndcg = []
    recall = []
    for user in users:
        scores = np.asarray(model.score(user), dtype=float).copy()
        if scores.ndim != 1 or len(scores) <= int(catalog.max()):
            raise ValueError('Model scores have the wrong catalog dimension')
        scores[list(seen.get(user, set()))] = -np.inf
        candidates = catalog[np.isfinite(scores[catalog])]
        if not len(candidates):
            raise ValueError(f'No unseen catalog items for evaluated user {user}')
        top = candidates[np.argsort(-scores[candidates], kind='stable')[:k]]
        relevant_items = relevant.get(user, set())
        if not relevant_items:
            ndcg.append(0.0)
            recall.append(0.0)
            continue
        hits = np.array([item in relevant_items for item in top], dtype=float)
        discounts = 1 / np.log2(np.arange(2, len(top) + 2))
        ideal_len = min(k, len(relevant_items))
        ideal = (1 / np.log2(np.arange(2, ideal_len + 2))).sum()
        ndcg.append(float((hits * discounts).sum() / ideal))
        recall.append(float(hits.sum() / len(relevant_items)))
    return {
        'ndcg@10': float(np.mean(ndcg)),
        'recall@10': float(np.mean(recall)),
        'users': len(ndcg),
    }
