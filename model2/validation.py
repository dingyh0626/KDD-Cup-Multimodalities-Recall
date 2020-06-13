import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from model import ScoreModel, ImageEncoder
# from model_v2 import TextEncoder
from utils import ValidDataset, collate_fn_valid, DataLoader
from tqdm import tqdm
import json
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'


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


def valid(epoch=1, checkpoints_dir='./checkpoints', use_bert=False, data_path=None, out_path='valid_pred2.json', output_ndcg=True):
    print("valid epoch{}".format(epoch))
    if data_path is not None:
        kdd_dataset = ValidDataset(data_path, use_bert=use_bert)
    else:
        kdd_dataset = ValidDataset(data_path, use_bert=use_bert)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn_valid, batch_size=128, shuffle=False, num_workers=3)
    tbar = tqdm(loader)
    # score_model = model.QueryEmbedding(kdd_dataset.unknown_token + 1, 768, use_bert=use_bert).cuda()
    # image_encoder = model.ItemEmbedding(2048, 768).cuda()
    nhead = 4
    score_model = ScoreModel(kdd_dataset.unknown_token + 1, 1024, 1024, use_bert=use_bert).cuda()
    image_encoder = ImageEncoder(input_dim=2048, output_dim=1024, nhead=nhead).cuda()
    checkpoints = torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch)))
    score_model.load_state_dict(checkpoints['query'])
    image_encoder.load_state_dict(checkpoints['score'])
    outputs = {}
    score_model.eval()
    image_encoder.eval()
    # score_model.eval()
    for query_id, product_id, query, query_len, features, boxes, category, obj_len in tbar:
        query, query_len = query.cuda(), query_len.cuda()

        features = image_encoder(features.cuda(), boxes.cuda(), obj_len.cuda())
        # score = score_func(query, features)
        score = score_model(query, query_len, features)
        score = score.data.cpu().numpy()

        # print(score2)

        for q_id, p_id, s in zip(query_id.data.numpy(), product_id.data.numpy(), score):
            outputs.setdefault(str(q_id), [])
            outputs[str(q_id)].append((p_id, s))

    for k, v in outputs.items():
        v = sorted(v, key=lambda x: x[1], reverse=True)
        v = [(str(x[0]), float(x[1])) for x in v]
        outputs[k] = v
    with open(out_path, 'w') as f:
        json.dump(outputs, f)

    pred = read_json(out_path)
    gt = read_json('/share/wulei/kdd-data/valid_answer.json')
    if output_ndcg:
        pred = read_json(out_path)
        gt = read_json('/share/wulei/kdd-data/valid_answer.json')
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
        print('ndcg@%d: %.4f' % (k, score))
        return score
    else:
        return None


if __name__ == '__main__':
    test_path = '/share/wulei/kdd-data/testB.tsv'
    valid_path = '/share/wulei/kdd-data/valid.tsv'
    checkpoints_dir = '/data/data_dyh/kdd_ckpt/ckpt_main/checkpoints3'
    epoch = 2
    # output validation prediction
    valid(epoch, checkpoints_dir, use_bert=True,
          data_path=valid_path, out_path='valid2_pred.json', output_ndcg=True)
    # output testing prediction
    valid(epoch, checkpoints_dir, use_bert=True,
          data_path=test_path, out_path='test2_pred.json', output_ndcg=False)
