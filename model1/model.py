import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
pretrained_large = 'bert-large-uncased'
pretrained_base = 'bert-base-uncased'

def lengths2mask(lengths, max_len):
    mask = torch.arange(max_len).to(lengths.device)
    mask = mask.unsqueeze_(0) >= lengths.unsqueeze(1)
    return mask



class ImageEncoder(nn.Module):
    def __init__(self, input_dim=2048, output_dim=1024, nhead=4, layers=1):
        super(ImageEncoder, self).__init__()
        self.box_encoding = nn.Sequential(
            nn.Linear(5, 100),
            nn.PReLU(),
            nn.Linear(100, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        self.freeze(self.box_encoding)
        self.freeze(self.norm)
        # self.encoder = nn.MultiheadAttention(input_dim, 8, dropout=0.01)
        self.encoders = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.ffs = nn.ModuleList()



        for i in range(layers):
            self.encoders.append(nn.MultiheadAttention(input_dim, 16, dropout=0.1))
            self.activations.append(nn.PReLU())
            self.ffs.append(nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.PReLU(),
                nn.Linear(4096, 2048),
                nn.PReLU()
            ))
        self.freeze(self.encoders)
        self.freeze(self.activations)
        self.freeze(self.ffs)
        self.nhead = nhead
        # self.relu = nn.ReLU(inplace=True)

        self.dense_summary = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
        )
        # self.dense_summary = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.PReLU(),
        #     nn.Linear(512, 1),
        # )
        # self.freeze(self.dense_summary)

        # self.dense_hidden = nn.Sequential(
        #     nn.BatchNorm1d(output_dim),
        #     nn.PReLU(),
        #     nn.Linear(output_dim, output_dim)
        # )
        output_dim *= (nhead + 1)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.PReLU(),
            nn.Linear(output_dim, output_dim)
        )
        # self.freeze(self.dense)
        # self.freeze(self)
        # for p in self.parameters():
        #     p.requires_grad_(False)

    def freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    def get_params(self):
        return [
            {'params': self.dense.parameters(), 'initial_lr': 5e-6, 'weight_decay': 1e-6},
            {'params': self.dense_summary.parameters(), 'initial_lr': 5e-6, 'weight_decay': 1e-6}
            # {'params': self.parameters(), 'initial_lr': 1e-4, 'weight_decay': 1e-6}
        ]

    # def get_params(self):
    #     return [
    #         {'params': self.box_encoding.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.encoders.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.ffs.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.dense.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.dense_summary.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.activations.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
    #         {'params': self.dense_hidden.parameters(), 'initial_lr': 1e-3},
    #     ]

    def load_pretrained_weights(self, path='/data/data_dyh/kdd_ckpt/ckpt_clf/checkpoints/image_encoder_large.pth'):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt, strict=False)

    def forward_multiheadattenton(self, encoder, ff, activation, x, mask):
        identity = x
        x, _ = encoder(x, x, x, key_padding_mask=mask)
        x = identity + x
        x = activation(x)
        x = ff(x)
        return x

    def forward(self, x, boxes, lengths):
        mask = lengths2mask(lengths, x.size(1))
        boxes = self.box_encoding(boxes)
        x = x + boxes
        x = self.norm(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        for encoder, ff, activation in zip(self.encoders, self.ffs, self.activations):
            x = self.forward_multiheadattenton(encoder, ff, activation, x, mask)
        x = x.permute(1, 0, 2)
        # torch.cat([x, boxes], -1)
        summary_score = self.dense_summary(x).squeeze_(-1)
        summary_score.masked_fill_(mask, -float('inf'))
        summary_score = torch.softmax(summary_score, -1)
        embedding = self.dense(x)
        embedding = torch.einsum('bid,bi->bd', [embedding, summary_score])

        embedding = embedding.view(embedding.size(0), self.nhead + 1, -1)

        # embedding = torch.cat([self.dense_hidden(embedding[:, 0, :]).unsqueeze_(1), embedding[:, 1:, :]], 1)
        # embedding = embedding[:, 1:, :]
        # batch, n_regions, dim = x.size()
        return embedding



class ScoreModel(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dim=256):
        super(ScoreModel, self).__init__()
        self.dense_summary = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, 512),
            nn.PReLU(),
            nn.Linear(512, 1)
        )
        self.start_token = 101
        self.end_token = 102

    def get_params(self):
        return [
            # {'params': self.embed.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
            # {'params': self.norm.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
            # {'params': self.rnn.parameters(), 'initial_lr': 1e-4},
            {'params': self.dense_summary.parameters(), 'initial_lr': 1e-4},
            # {'params': self.dense_hidden.parameters(), 'initial_lr': 1e-3},
        ]

    def forward(self, txt_embedding, txt_hidden, txt_lengths, img_embedding, cross=False):
        mask = lengths2mask(txt_lengths, txt_hidden.size(1))
        # mask.data[:, 0] = True
        # mask.data.scatter_(1, txt_lengths.unsqueeze(1) - 1, True)
        img_hidden = img_embedding[:, 0, :].contiguous()
        img_embedding = img_embedding[:, 1:, :]

        batch1, max_len, dim1 = txt_hidden.size()
        batch2, dim2 = img_hidden.size()

        if cross:
            txt_hidden = txt_hidden.unsqueeze(1)
            img_hidden = img_hidden.unsqueeze(0).unsqueeze(-2)
            txt_hidden = txt_hidden.repeat(1, batch2, 1, 1)
            img_hidden = img_hidden.repeat(batch1, 1, max_len, 1)
            summary_score = self.dense_summary(torch.cat([txt_hidden, img_hidden], -1)).squeeze_(-1)
            mask = mask.unsqueeze_(1)
            summary_score.data.masked_fill_(mask, -float('inf'))
            summary_score = torch.softmax(summary_score, -1)
            score = torch.einsum('mid,nhd->mnhi', [txt_embedding, img_embedding])
            score = score.max(-2)[0]
            score = score * summary_score
            score = score.sum(-1)
        else:
            img_hidden = img_hidden.unsqueeze(-2).repeat(1, max_len, 1)
            summary_score = self.dense_summary(torch.cat([txt_hidden, img_hidden], -1)).squeeze_(-1)
            summary_score.data.masked_fill_(mask, -float('inf'))
            summary_score = torch.softmax(summary_score, -1)
            score = torch.einsum('bid,bhd->bhi', [txt_embedding, img_embedding])
            score = score.max(-2)[0]
            score = score * summary_score
            score = score.sum(-1)
        # if not self.training:
        #     return score, score_word
        return score



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0., max_violation=False, reduction='mean', bce=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.reduction = reduction
        self.bce = bce

    def forward(self, scores):
        '''
        :param scores: [n, n] The diagonal elements are the positive pairs.
        :return:
        '''
        # compute image-sentence score matrix
        n = scores.size(0)
        diagonal = scores.diag()


        # compare every diagonal score to scores in its column
        # caption retrieval
        # nn.MarginRankingLoss
        mask = torch.eye(n).bool()
        I = mask.to(scores.device)
        if not self.bce:
            diagonal = diagonal.view(n, 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

            # keep the maximum violating negative for each query
            if self.max_violation:
                cost_s = cost_s.max(1)[0]
                cost_im = cost_im.max(0)[0]
                if self.reduction == 'mean':
                    return cost_s.mean() + cost_im.mean()
                return cost_s.sum() + cost_im.sum()
            else:
                if self.reduction == 'mean':
                    I = ~I
                    return cost_s.masked_select(I).mean() + cost_im.masked_select(I).mean()
                return cost_s.sum() + cost_im.sum()
        else:
            # cost_s = -(F.logsigmoid(diagonal))
            cost_pos = -F.logsigmoid(diagonal)
            cost_neg = -F.logsigmoid(-scores)
            cost_neg = cost_neg.masked_fill_(I, 0)
            if self.max_violation:
                return 2 * cost_pos.mean() + cost_neg.max(1)[0].mean() + cost_neg.max(0)[0].mean()
            else:
                I = ~I
                cost_neg = cost_neg.masked_select(I)
                return cost_pos.mean() + cost_neg.mean()


