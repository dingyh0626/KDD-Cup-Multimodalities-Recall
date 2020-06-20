from torch.utils.data import dataset
from torch.utils.data import DataLoader as DataLoader_, Sampler
import os
import pickle
import pandas as pd
import re
import random
import numpy as np
import torch
from tqdm import tqdm
import base64
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from glob import glob
from multiprocessing import Pool
from prefetch_generator import BackgroundGenerator
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from nltk.corpus import stopwords
tokenizer_class, pretrained_weights = BertTokenizerFast, 'bert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

stop_words = stopwords.words('english')
stop_words += ['-', "'", 's'] + ['[UNK]', '[PAD]', '[MASK]']  #+ list(tokenizer.special_tokens_map.values()) # + [tokenizer.cls_token, tokenizer.sep_token]


class DataLoader(DataLoader_):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=10)


def _read(path):
    product_id = int(os.path.basename(path).split('.')[0])
    with open(path, 'rb') as f:
        data = np.load(f)
        data = {'boxes': data['boxes'], 'features': data['features']}
    return product_id, data


def process_category(category_label, label_list, use_bert=False, lexicon=None, unknown_token=None):
    ret = []
    if use_bert:
        max_ = 30
    else:
        max_ = 9
    length = []
    for l in label_list:
        # sent = tokenizer.encode(category_label.loc[l, 'category_name'])
        sent = encode_text(category_label.loc[l, 'category_name'], use_bert=use_bert, lexicon=lexicon, unknown_token=unknown_token, pattern='[- ().,/]')
        length.append(len(sent))
        ret.append(sent)
    ret = [sent + [0] * (max_ - len(sent)) for sent in ret]
    return np.array(ret), np.array(length)


def encode_text(text, use_bert=False, lexicon=None, unknown_token=None, pattern='[- ]'):
    if use_bert:

        text = tokenizer.tokenize(text)
        text = [t for t in text if t not in stop_words]

        return tokenizer.convert_tokens_to_ids(text)
    else:
        return [lexicon.get(w, 0) for w in re.split(pattern, text.lower()) if w != '']


class Dataset(dataset.Dataset):
    def __init__(self, root_dir='../data', use_bert=False):
        super(Dataset, self).__init__()
        self.use_bert = use_bert
        # pool = Pool(16)
        # products = pool.map(_read, glob(os.path.join(root_dir, 'features', '*.npz')))
        # pool.close()
        # pool.join()
        # self.products = dict(products)
        # self.products = {}
        # for p in tqdm(glob(os.path.join(root_dir, 'features', '*.npz'))):
        #     # if random.random() > 0.4:
        #     product_id = os.path.basename(p).split('.')[0]
        #     with open(p, 'rb') as f:
        #         self.products.setdefault(int(product_id), np.load(f))
        # print(len(self.products))

        # with open(os.path.join(root_dir, 'features.pkl'), 'rb') as f:
        #     self.products = pickle.load(f)

        self.root_dir = root_dir
        # data = []
        # for i in range(6):
        #     with open(os.path.join(root_dir, 'data_index_%d.pkl' % i), 'rb') as f:
        #         data += pickle.load(f)
        # data = pd.DataFrame(data, columns=['product_id', 'image_h', 'image_w', 'num_boxes', 'class_labels', 'query',
        #                                    'query_id'])
        # data.set_index('product_id', inplace=True)
        with open(os.path.join(root_dir, 'info', 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(root_dir, 'info', 'data_info2.pkl'), 'rb') as f:
            map_info = pickle.load(f)
        self.lexicon = map_info['lexicon']
        self.unknown_token = max(self.lexicon.values()) + 1
        self.label_map_query = map_info['label_map_query']
        self.label_map_product = map_info['label_map_product']
        for l, s in self.label_map_product.items():
            self.label_map_product[l] = list(s)

        for l, s in self.label_map_query.items():
            self.label_map_query[l] = list(s)
        self.data = data
        # self.query = pd.read_pickle(os.path.join(root_dir, 'query2product2.pkl'))
        self.cluster2query_id = map_info['cluster2query_id']
        self.last2query_id = map_info['last2query']
        self.last2product_id = map_info['last2product']
        self.category_label = pd.read_csv(os.path.join(root_dir, 'multimodal_labels.txt'), sep='\t')
        self.category_label.set_index('category_id', inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        product_id = row.name
        query = row['query']
        query = encode_text(query, self.use_bert, self.lexicon, self.unknown_token)
        # if self.use_bert:
        #     query = query[1:-1]

        feature_product = np.load(os.path.join(self.root_dir, 'features', '{}.npz'.format(product_id)))
        features = feature_product['features']
        obj_len = len(features)
        query_len = len(query)

        boxes_ = feature_product['boxes']
        h, w = row['image_h'], row['image_w']
        boxes = np.zeros((len(boxes_), 5))
        boxes[:, 0] = boxes_[:, 0] / h
        boxes[:, 2] = boxes_[:, 2] / h
        boxes[:, 1] = boxes_[:, 1] / w
        boxes[:, 3] = boxes_[:, 3] / w
        boxes[:, 4] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return query, query_len, features, boxes, obj_len




class ValidDataset(dataset.Dataset):
    def __init__(self, path_tsv='../data/valid/valid.tsv', path_map_info='../data/info/data_info2.pkl', use_bert=False):
        super(ValidDataset, self).__init__()
        with open(os.path.join(path_map_info), 'rb') as f:
            map_info = pickle.load(f)
        self.lexicon = map_info['lexicon']
        self.tag = map_info['tag']
        self.unknown_token = max(self.lexicon.values()) + 1
        self.data = pd.read_csv(path_tsv, sep='\t')
        # self.category_label = pd.read_csv(path_label, sep='\t')
        # self.category_label.set_index('category_id', inplace=True)
        self.use_bert = use_bert

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        query = row['query']
        query_id = row['query_id']
        product_id = row['product_id']
        # query = [self.lexicon.get(w, self.unknown_token) for w in re.split('[- ]', query.lower())]

        query = encode_text(query, self.use_bert, self.lexicon, self.unknown_token)
        # if self.use_bert:
        #     query = query[1:-1]
        query = np.asarray(query)
        query_len = len(query)

        features = np.frombuffer(base64.b64decode(row['features']), np.float32).reshape((-1, 2048))
        obj_len = len(features)

        boxes_ = np.frombuffer(base64.b64decode(row['boxes']), np.float32).reshape((-1, 4))
        h, w = row['image_h'], row['image_w']
        boxes = np.zeros((len(boxes_), 5))
        boxes[:, 0] = boxes_[:, 0] / h
        boxes[:, 2] = boxes_[:, 2] / h
        boxes[:, 1] = boxes_[:, 1] / w
        boxes[:, 3] = boxes_[:, 3] / w
        boxes[:, 4] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return query_id, product_id, query, query_len, features, boxes, obj_len


def collate_fn_valid(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_query_id, batch_product_id, batch_query, batch_query_len, batch_features, batch_boxes, batch_obj_len = [], [], [], [], [], [], []
    max_query_len = max([b[3] for b in batch])
    max_obj_len = max([b[-1] for b in batch])
    for query_id, product_id, query, query_len, features, boxes, obj_len in batch:

        batch_query_id.append(query_id)
        batch_product_id.append(product_id)

        query = np.concatenate([query, np.zeros(max_query_len - len(query), dtype=np.int)])
        batch_query.append(query)
        batch_query_len.append(query_len)

        features = np.concatenate([features, np.zeros((max_obj_len - len(features), 2048))])
        batch_features.append(features)

        boxes = np.concatenate([boxes, np.zeros((max_obj_len - len(boxes), 5))])
        batch_boxes.append(boxes)



        batch_obj_len.append(obj_len)

    query_id = torch.from_numpy(np.stack(batch_query_id, 0)).long()
    product_id = torch.from_numpy(np.stack(batch_product_id, 0)).long()

    query = torch.from_numpy(np.stack(batch_query, 0)).long()
    query_len = torch.from_numpy(np.stack(batch_query_len, 0)).long()


    features = torch.from_numpy(np.stack(batch_features, 0)).float()
    obj_len = torch.from_numpy(np.stack(batch_obj_len, 0)).long()

    boxes = torch.from_numpy(np.stack(batch_boxes, 0)).float()
    return query_id, product_id, query, query_len, features, boxes, obj_len


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_query, batch_query_len, batch_features, batch_boxes, batch_obj_len = [], [], [], [], []
    max_query_len = max([b[1] for b in batch])
    max_obj_len = max([b[-1] for b in batch])
    for query, query_len, features, boxes, obj_len in batch:
        query = np.concatenate([query, np.zeros(max_query_len - len(query), dtype=np.int)])
        batch_query.append(query)
        batch_query_len.append(query_len)

        features = np.concatenate([features, np.zeros((max_obj_len - len(features), 2048))])
        batch_features.append(features)

        boxes = np.concatenate([boxes, np.zeros((max_obj_len - len(boxes), 5))])
        batch_boxes.append(boxes)

        batch_obj_len.append(obj_len)

    query = torch.from_numpy(np.stack(batch_query, 0)).long()
    query_len = torch.from_numpy(np.stack(batch_query_len, 0)).long()
    boxes = torch.from_numpy(np.stack(batch_boxes, 0)).float()
    features = torch.from_numpy(np.stack(batch_features, 0)).float()
    obj_len = torch.from_numpy(np.stack(batch_obj_len, 0)).long()
    return query, query_len, features, boxes, obj_len


class CustomSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source: Dataset, batch_size):
        super(CustomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.df_query = data_source.query
        self.batch_size = batch_size
        self.cluster2query_id = data_source.cluster2query_id
        self.product_id2index = dict(zip(data_source.data.index, range(len(data_source))))
        self.cluster = None
        self._cluster_list = list(self.cluster2query_id.keys())
        self.query_id_set = set()


    def __iter__(self):
        # return iter(range(len(self.data_source)))
        for i in range(len(self.data_source)):
            if i % self.batch_size == 0:
                self.cluster = random.choice(self._cluster_list)
                self.query_id_set = set()
            cluster = self.cluster2query_id[self.cluster]
            while True:
                query_id = random.choice(cluster)
                if query_id not in self.query_id_set:
                    break
            self.query_id_set.add(query_id)
            product_id = random.choice(self.df_query.loc[query_id, 'product_id'])
            yield self.product_id2index[product_id]

    def __len__(self):
        return len(self.data_source)


if __name__ == '__main__':
    kdd_dataset = ValidDataset(use_bert=True)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn_valid, batch_size=128, shuffle=False, num_workers=8)
    for query_id, product_id, query, query_len, features, obj_len in loader:
        break