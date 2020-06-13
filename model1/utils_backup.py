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

# import json
# with open('info/tokens.json', 'r') as f:
#     tokens = json.load(f)

# tokenizer.add_tokens(tokens)
stop_words = stopwords.words('english')
stop_words += ['-', "'", 's']


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
        sent = encode_text(category_label.loc[l, 'category_name'], use_bert=use_bert, lexicon=lexicon,
                           unknown_token=unknown_token, pattern='[- ().,/]')
        length.append(len(sent))
        ret.append(sent)
    ret = [sent + [0] * (max_ - len(sent)) for sent in ret]
    return np.array(ret), np.array(length)


def encode_text(text, use_bert=False, lexicon=None, unknown_token=None, pattern='[- ]'):
    if use_bert:
        text = tokenizer.tokenize(text)
        text = [t for t in text if t not in stop_words]
        return tokenizer.convert_tokens_to_ids(text)
        # return tokenizer.encode(text)
    else:
        return [lexicon.get(w, 0) for w in re.split(pattern, text.lower()) if w != '']


class Dataset(dataset.Dataset):
    def __init__(self, root_dir='/home/dingyuhui/dataset/kdd-data', use_bert=False):
        super(Dataset, self).__init__()
        self.use_bert = use_bert

        self.root_dir = root_dir
        data = pd.read_csv(os.path.join(root_dir, 'train.tsv'), sep='\t')
        data.set_index('product_id', inplace=True)
        with open(os.path.join(root_dir, 'data_info2.pkl'), 'rb') as f:
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
        self.query = pd.read_pickle(os.path.join(root_dir, 'query2product2.pkl'))
        self.cluster2query_id = map_info['cluster2query_id']
        self.last2query_id = map_info['last2query']
        self.last2product_id = map_info['last2product']
        self.category_label = pd.read_csv(os.path.join(root_dir, 'multimodal_labels.txt'), sep='\t')
        self.category_label.set_index('category_id', inplace=True)

    def __len__(self):
        return len(self.data)

    def read_product(self, row):
        features = np.frombuffer(base64.b64decode(row['features']), np.float32).reshape((-1, 2048))
        boxes = np.frombuffer(base64.b64decode(row['boxes']), np.float32).reshape((-1, 4))
        return {
            'features': features,
            'boxes': boxes
        }

    def __getitem__(self, i):
        row = self.data.iloc[i]
        product_id = row.name
        query = row['query']
        query_id = row['query_id']
        query = encode_text(query, self.use_bert, self.lexicon, self.unknown_token)

        # query = [self.lexicon.get(w, self.unknown_token) for w in re.split('[- ]', query.lower())]
        # while True:
        #     idx = random.choice(range(len(self)))
        #     row_ = self.data.iloc[idx]
        #     query_id_negative = row_['query_id']
        #     if query_id_negative != query_id:
        #         query_negative = tokenizer.encode(row_['query'])
        #         h_negative, w_negative = row_['image_h'], row_['image_w']
        #         # query_negative = [self.lexicon.get(w, self.unknown_token) for w in re.split('[- ]', row_['query'].lower())]
        #         break

        # category, category_len = process_category(self.category_label, class_labels, self.use_bert, self.lexicon, self.unknown_token)
        query_row = self.query.loc[query_id]
        last_word = query_row['last_word']
        cluster = query_row['cluster']
        neg_query_id_set = list(
            set(self.last2query_id[last_word]).difference([query_id])) if random.random() > 0.2 else []
        # neg_query_id_set = []
        while True:
            # query_id_negative = random.choice(self.label_map_query[random.choice(class_labels)])
            if len(neg_query_id_set) > 0:
                query_id_negative = random.choice(neg_query_id_set)
            else:
                query_id_negative = random.choice(
                    self.cluster2query_id[cluster]) if random.random() > 0.5 else random.choice(self.query.index)
            if query_id_negative != query_id:
                query_negative = self.query.loc[query_id_negative, 'query']
                query_negative = encode_text(query_negative, self.use_bert, self.lexicon, self.unknown_token)
                # category = [int(x) + 1 for x in class_labels]
                # category = query_row['tag']
                # query_negative = [self.lexicon.get(w, self.unknown_token) for w in re.split('[- ]', query_negative.lower())]
                break

        query = np.asarray(query)
        query_negative = np.asarray(query_negative)
        query_len = len(query)
        query_negative_len = len(query_negative)

        product_positive_list = list(query_row['product_id'])
        neg_query_id_set = list(
            set(self.last2query_id[last_word]).difference([query_id])) if random.random() > 0.2 else []
        while True:
            # product_id_negative = random.choice(self.label_map_product[random.choice(class_labels)])
            if len(neg_query_id_set) > 0:
                query_row = self.query.loc[random.choice(neg_query_id_set)]
            else:
                id_ = random.choice(self.cluster2query_id[cluster]) if random.random() > 0.5 else random.choice(
                    self.query.index)
                query_row = self.query.loc[id_]
            product_id_negative = random.choice(query_row['product_id'])
            # product_id_negative = random.choice(self.last2product_id[last_word])
            if product_id_negative not in product_positive_list:
                row_ = self.data.loc[product_id_negative]
                h_negative, w_negative = row_['image_h'], row_['image_w']
                # category_negative = [int(x) + 1 for x in row_['class_labels']]
                # category_negative = query_row['tag']
                # category_negative, category_negative_len = process_category(self.category_label, row_['class_labels'], self.use_bert, self.lexicon, self.unknown_token)
                break


        feature_product = self.read_product(row)
        feature_product_negative = self.read_product(row_)

        boxes_ = feature_product['boxes']
        h, w = row['image_h'], row['image_w']
        boxes = np.zeros((len(boxes_), 5))
        boxes[:, 0] = boxes_[:, 0] / h
        boxes[:, 2] = boxes_[:, 2] / h
        boxes[:, 1] = boxes_[:, 1] / w
        boxes[:, 3] = boxes_[:, 3] / w
        boxes[:, 4] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        features = feature_product['features']
        obj_len = len(boxes)

        boxes_negative_ = feature_product_negative['boxes']
        boxes_negative = np.zeros((len(boxes_negative_), 5))
        boxes_negative[:, 0] = boxes_negative_[:, 0] / h_negative
        boxes_negative[:, 2] = boxes_negative_[:, 2] / h_negative
        boxes_negative[:, 1] = boxes_negative_[:, 1] / w_negative
        boxes_negative[:, 3] = boxes_negative_[:, 3] / w_negative
        boxes_negative[:, 4] = (boxes_negative[:, 3] - boxes_negative[:, 1]) * (
                    boxes_negative[:, 2] - boxes_negative[:, 0])
        features_negative = feature_product_negative['features']
        obj_negative_len = len(boxes_negative)

        return query_id, product_id, query, query_len, features, boxes, obj_len, \
               query_negative, query_negative_len, features_negative, boxes_negative, obj_negative_len


class ValidDataset(dataset.Dataset):
    def __init__(self, path_tsv='/share/wulei/kdd-data/valid.tsv',
                 path_map_info='/home/dingyuhui/dataset/kdd-data/data_info2.pkl',
                 path_label='/home/dingyuhui/dataset/kdd-data/multimodal_labels.txt', use_bert=False):
        super(ValidDataset, self).__init__()
        with open(os.path.join(path_map_info), 'rb') as f:
            map_info = pickle.load(f)
        self.lexicon = map_info['lexicon']
        self.tag = map_info['tag']
        self.unknown_token = max(self.lexicon.values()) + 1
        self.data = pd.read_csv(path_tsv, sep='\t')
        self.category_label = pd.read_csv(path_label, sep='\t')
        self.category_label.set_index('category_id', inplace=True)
        self.use_bert = use_bert

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        query = row['query']
        query_id = row['query_id']
        product_id = row['product_id']
        # query = [self.lexicon.get(w, self.unknown_token) for w in re.split('[- ]', query.lower())]
        tag = nltk.pos_tag(nltk.word_tokenize(query))
        tag = [w for w, p in tag if p[0] == 'J' or p[0] == 'N']
        tag = [lemmatizer.lemmatize(w) for w in tag]
        tag = list(set(tag))
        tag = [self.tag.get(t, 0) for t in tag]

        query = encode_text(query, self.use_bert, self.lexicon, self.unknown_token)
        query = np.asarray(query)
        query_len = len(query)

        features = np.frombuffer(base64.b64decode(row['features']), np.float32).reshape((-1, 2048))
        boxes_ = np.frombuffer(base64.b64decode(row['boxes']), np.float32).reshape((-1, 4))
        # class_labels = np.frombuffer(base64.b64decode(row['class_labels']), dtype=np.int64)
        # category, category_len = process_category(self.category_label, class_labels, self.use_bert, self.lexicon, self.unknown_token)
        # category = [int(x) + 1 for x in class_labels]
        h, w = row['image_h'], row['image_w']
        boxes = np.zeros((len(boxes_), 5))
        boxes[:, 0] = boxes_[:, 0] / h
        boxes[:, 2] = boxes_[:, 2] / h
        boxes[:, 1] = boxes_[:, 1] / w
        boxes[:, 3] = boxes_[:, 3] / w
        boxes[:, 4] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        obj_len = len(features)
        return query_id, product_id, query, query_len, features, boxes, tag, obj_len


def collate_fn_valid(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_query_id, batch_product_id, batch_query, batch_query_len, batch_features, batch_boxes, batch_category, batch_obj_len = [], [], [], [], [], [], [], []
    max_query_len = max([b[3] for b in batch])
    max_obj_len = max([b[-1] for b in batch])
    for query_id, product_id, query, query_len, features, boxes, category, obj_len in batch:
        batch_query_id.append(query_id)
        batch_product_id.append(product_id)
        query = np.concatenate([query, np.zeros(max_query_len - len(query), dtype=np.int)])
        batch_query.append(query)
        batch_query_len.append(query_len)

        boxes = np.concatenate([boxes, np.zeros((max_obj_len - len(boxes), 5))])
        batch_boxes.append(boxes)
        features = np.concatenate([features, np.zeros((max_obj_len - len(features), 2048))])
        batch_features.append(features)
        batch_obj_len.append(obj_len)

        category = np.concatenate([category, np.zeros((12 - len(category)), dtype=np.int)])
        batch_category.append(category)
        # category_len = np.concatenate([category_len, np.ones(max_obj_len - len(category_len), dtype=np.int)])
        # batch_category_len.append(category_len)

    query_id = torch.from_numpy(np.stack(batch_query_id, 0)).long()
    product_id = torch.from_numpy(np.stack(batch_product_id, 0)).long()
    query = torch.from_numpy(np.stack(batch_query, 0)).long()
    query_len = torch.from_numpy(np.stack(batch_query_len, 0)).long()

    boxes = torch.from_numpy(np.stack(batch_boxes, 0)).float()
    features = torch.from_numpy(np.stack(batch_features, 0)).float()
    obj_len = torch.from_numpy(np.stack(batch_obj_len, 0)).long()
    category = torch.from_numpy(np.stack(batch_category, 0)).long()
    # category_len = torch.from_numpy(np.stack(batch_category_len, 0)).long()
    return query_id, product_id, query, query_len, features, boxes, category, obj_len


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_query, batch_query_len, batch_features, batch_boxes, batch_obj_len, \
    batch_query_negative, batch_query_negative_len, batch_features_negative, batch_boxes_negative, batch_obj_negative_len = [], [], [], [], [], [], [], [], [], []
    # batch_query, batch_query_len, batch_boxes, batch_features, batch_obj_len = [], [], [], [], []
    max_query_len = max([b[3] for b in batch])
    max_obj_len = max([b[6] for b in batch])
    max_query_negative_len = max([b[8] for b in batch])
    max_obj_negative_len = max([b[-1] for b in batch])
    query_set = set()
    for query_id, product_id, query, query_len, features, boxes, obj_len, \
        query_negative, query_negative_len, features_negative, boxes_negative, obj_negative_len in batch:
        if query_len == 0 or query_negative_len == 0:
            continue
        if query_id in query_set:
            continue
        query_set.add(query_id)
        query = np.concatenate([query, np.zeros(max_query_len - len(query), dtype=np.int)])
        batch_query.append(query)
        batch_query_len.append(query_len)

        query_negative = np.concatenate(
            [query_negative, np.zeros(max_query_negative_len - len(query_negative), dtype=np.int)])
        batch_query_negative.append(query_negative)
        batch_query_negative_len.append(query_negative_len)

        boxes = np.concatenate([boxes, np.zeros((max_obj_len - len(boxes), 5))])
        batch_boxes.append(boxes)

        # category = np.concatenate([category, np.zeros(12 - len(category), dtype=np.int)])
        # batch_category.append(category)

        # category = np.concatenate([category, np.zeros((max_obj_len - len(category), category.shape[1]))])
        # batch_category.append(category)

        # category_len = np.concatenate([category_len, np.ones(max_obj_len - len(category_len))])
        # batch_category_len.append(category_len)

        features = np.concatenate([features, np.zeros((max_obj_len - len(features), 2048))])
        batch_features.append(features)
        batch_obj_len.append(obj_len)

        boxes_negative = np.concatenate([boxes_negative,
                                         np.zeros((max_obj_negative_len - len(boxes_negative), 5))])
        batch_boxes_negative.append(boxes_negative)

        # category_negative = np.concatenate([category_negative, np.zeros(12 - len(category_negative), dtype=np.int)])
        # batch_category_negative.append(category_negative)

        # category_negative = np.concatenate([category_negative, np.zeros((max_obj_negative_len - len(category_negative), category_negative.shape[1]))])
        # batch_category_negative.append(category_negative)
        # category_negative_len = np.concatenate([category_negative_len, np.ones(max_obj_negative_len - len(category_negative_len))])
        # batch_category_negative_len.append(category_negative_len)

        features_negative = np.concatenate(
            [features_negative, np.zeros((max_obj_negative_len - len(features_negative), 2048))])
        batch_features_negative.append(features_negative)
        batch_obj_negative_len.append(obj_negative_len)

    query = torch.from_numpy(np.stack(batch_query, 0)).long()
    query_len = torch.from_numpy(np.stack(batch_query_len, 0)).long()

    query_negative = torch.from_numpy(np.stack(batch_query_negative, 0)).long()
    query_negative_len = torch.from_numpy(np.stack(batch_query_negative_len, 0)).long()

    boxes = torch.from_numpy(np.stack(batch_boxes, 0)).float()
    features = torch.from_numpy(np.stack(batch_features, 0)).float()
    # category = torch.from_numpy(np.stack(batch_category, 0)).long()
    # category = torch.from_numpy(np.stack(batch_category, 0)).long()
    # category_len = torch.from_numpy(np.stack(batch_category_len, 0)).long()
    obj_len = torch.from_numpy(np.stack(batch_obj_len, 0)).long()

    boxes_negative = torch.from_numpy(np.stack(batch_boxes_negative, 0)).float()
    features_negative = torch.from_numpy(np.stack(batch_features_negative, 0)).float()
    # category_negative = torch.from_numpy(np.stack(batch_category_negative, 0)).long()
    # category_negative_len = torch.from_numpy(np.stack(batch_category_negative_len, 0)).long()
    # category_negative = torch.from_numpy(np.stack(batch_category_negative, 0)).long()
    obj_negative_len = torch.from_numpy(np.stack(batch_obj_negative_len, 0)).long()

    return query, query_len, features, boxes, obj_len, \
           query_negative, query_negative_len, features_negative, boxes_negative, obj_negative_len


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
    # pool = Pool(16)
    # root_dir = '/home/dingyuhui/dataset/kdd-data'
    # products = pool.map(_read, glob(os.path.join(root_dir, 'features', '*.npz')))
    # pool.close()
    # pool.join()
    # products = dict(products)
    # with open(os.path.join(root_dir, 'features.pkl'), 'wb') as f:
    #     pickle.dump(products, f)
    # with open('error.log', 'w') as f:
    #     for p in tqdm(glob(os.path.join('/home/dingyuhui/dataset/kdd-data/features', '*.npz'))):
    #         with open(p, 'rb') as fp:
    #             data = np.load(fp)
    #         product_id = os.path.basename(p).split('.')[0]
    kdd_dataset = Dataset()
    sampler = CustomSampler(kdd_dataset, 100)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn, batch_size=100, sampler=sampler, num_workers=20)
    for query, query_len, features, boxes, category, obj_len, \
        query_negative, query_negative_len, features_negative, boxes_negative, category_negative, obj_negative_len in tqdm(
        loader):
        pass
        # print(query_id.size())
        # print(query.size())
        # print(query_len.size())
        # print(boxes.size())
        # print(features.size())
        # print(obj_len.size())
        # break
