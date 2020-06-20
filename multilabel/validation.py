import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from utils import ValidDataset, DataLoader, collate_fn_valid
from multilabel import MultiLabelClassifier
import numpy as np
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast
tokenizer_class, pretrained_weights = BertTokenizerFast, 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# from bert_score import BERTScorer

# scorer = BERTScorer(model_type='bert-base-uncased', lang='en')


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


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def decode(idx):
    text = tokenizer.decode(idx, clean_up_tokenization_spaces=False).split('[SEP]')[0]
    return text
    # ret = []
    # # for t in text:
    # #     if t == '[SEP]':
    # #         break
    # #     ret.append(t)
    # return ' '.join(ret)




def valid(epoch=1, checkpoints_dir='./checkpoints', use_bert=False, large=False):
    print("valid epoch{}".format(epoch))
    kdd_dataset = ValidDataset(use_bert=use_bert)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn_valid, batch_size=128, shuffle=False, num_workers=8)
    # tbar = tqdm(loader)
    # query_embedding = model.QueryEmbedding(kdd_dataset.unknown_token + 1, 256, use_bert=use_bert).cuda()
    # item_embedding = model.ItemEmbedding(2048, 256).cuda()
    model = MultiLabelClassifier(large).cuda()
    checkpoints = torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch)))

    model.load_state_dict(checkpoints['model'])
    model.eval()
    outputs = {}
    for query_id, product_id, query, query_len, features, boxes, obj_len in loader:
        query = query.cuda()
        features = features.cuda()
        obj_len = obj_len.cuda()
        boxes = boxes.cuda()
        with torch.autograd.no_grad():
            index, loss, caps = model(features, boxes, obj_len, query)

        score = -loss
        score = score.data.cpu().numpy()
        # print(score2)

        for text, q_id, p_id, s, idx, sents in zip(query.data.cpu().numpy(), query_id.data.numpy(), product_id.data.numpy(),
                                            score, index.data.cpu().numpy(), caps.data.cpu().numpy()):
            # ref_vec = [1.0] * len(text)
            # pred_vec = [1.0 if pid in text else 0.0 for pid in idx]
            # # s = get_ndcg(pred_vec, ref_vec, k=5)
            # s = dcg_at_k(pred_vec, 15)





            print(p_id)
            # # ref = [decode(sent)]
            # # cand = [decode(text)]
            # # print(decode(text))
            for t in sents:
                print(decode(t))
            # P, R, F1 = scorer.score(cand, ref)
            # s = F1.item()
            outputs.setdefault(str(q_id), [])
            outputs[str(q_id)].append((p_id, s))

            # print(tokenizer.decode(idx))

    for k, v in outputs.items():
        v = sorted(v, key=lambda x: x[1], reverse=True)
        v = [str(x[0]) for x in v]
        outputs[k] = v

    with open('../prediction_result/valid_pred.json', 'w') as f:
        json.dump(outputs, f)

    pred = read_json('../prediction_result/valid_pred.json')
    gt = read_json('../data/valid/kdd-data/valid_answer.json')
    score = 0
    k = 5
    for key, val in gt.items():
        ground_truth_ids = [str(x) for x in val]
        predictions = pred[key][:k]
        ref_vec = [1.0] * len(ground_truth_ids)

        pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions]
        score += get_ndcg(pred_vec, ref_vec, k)
        # print(key)
        # print([pid for pid in predictions if pid not in ground_truth_ids])
        # print('========')
        # score += len(set(predictions).intersection(ground_truth_ids)) / len(ground_truth_ids)
    print('ndcg@%d: %.4f' % (k, score / len(gt)))

if __name__ == '__main__':
    # print(get_ndcg([0, 0, 0, 1, 1], [1, 1, 1, 1, 1], 5))
    # valid(10, checkpoints_dir='./checkpoints3', use_bert=True)
    valid(6, checkpoints_dir='./checkpoints_large', use_bert=True, large=True)







