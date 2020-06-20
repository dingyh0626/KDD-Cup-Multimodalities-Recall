
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd


# In[2]:


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# In[3]:


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.


# compute ndcg@k (dcg@k / idcg@k) for a single sample
def get_ndcg(r, ref, k):
    dcg_max = dcg_at_k(ref, k)
    if not dcg_max:
        return 0.
    dcg = dcg_at_k(r, k)
    return dcg / dcg_max


# #### Read validation prediction from the two models

# In[4]:


pred1 = read_json('prediction_result/valid_pred_model1.json')
pred2 = read_json('prediction_result/valid_pred_model2.json')
gt = read_json('./data/valid/valid_answer.json')


# #### Search for the best threshold `t` to ensemble the two models

# In[5]:


best_score = 0
t = 0
for i in np.linspace(0, 1, 100):
    pred = {}
    for k in pred1.keys():
        v1 = pred1[k]
        v2 = pred2[k]
        v1 = dict(v1)
        v2 = dict(v2)
        v = []
        for kk in v1.keys():
            v.append((kk, i * v1[kk] + (1 - i) * v2[kk]))
    #     v = [(v2[i][0], v1[i][1] + v2[i][1]) for i in range(len(v1))]
        v = sorted(v, key=lambda x: x[1], reverse=True)
        pred[k] = v
        
    score = 0
    k = 5
    for key, val in gt.items():
        ground_truth_ids = [str(x) for x in val]
        predictions = [x[0] for x in pred[key][:k]]
        ref_vec = [1.0] * len(ground_truth_ids)

        pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions]
        score += get_ndcg(pred_vec, ref_vec, k)
        # print(key)
        # print([pid for pid in predictions if pid not in ground_truth_ids])
        # print('========')
        # score += len(set(predictions).intersection(ground_truth_ids)) / len(ground_truth_ids)
    score = score / len(gt)
    if score > best_score:
        best_score = score
        t = i
        print('best score: %.4f, best t: %.4f' % (score, i))
#     print('ndcg@%d: %.4f' % (k, score / len(gt)))


# #### Read testing prediction from the two models

# In[6]:


pred1 = read_json('prediction_result/test_pred_model1.json')
pred2 = read_json('prediction_result/test_pred_model2.json')


# #### Calculate the ensembled score for the testing data and output submission

# In[7]:


pred = {}
for k in pred1.keys():
    v1 = pred1[k]
    v2 = pred2[k]
    v1 = dict(v1)
    v2 = dict(v2)
    v = []
    for kk in v1.keys():
        v.append((kk, t * v1[kk] + (1 - t) * v2[kk]))
#     v = [(v2[i][0], v1[i][1] + v2[i][1]) for i in range(len(v1))]
    v = sorted(v, key=lambda x: x[1], reverse=True)
    pred[k] = v

submission = []
for k, v in pred.items():
    v = sorted(v, key=lambda x: x[1], reverse=True)
    v = [x[0] for x in v]
    submission.append([k] + v[:5])

submission = pd.DataFrame(submission, columns=['query-id', 'product1', 'product2',
                                               'product3', 'product4', 'product5'])
submission.to_csv('prediction_result/submission.csv', index=False)

