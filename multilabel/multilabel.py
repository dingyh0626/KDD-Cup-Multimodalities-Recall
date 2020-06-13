import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM
import torch.nn.functional as F
from torch.autograd import Variable

pretrained_base = 'bert-base-uncased'
pretrained_large = 'bert-large-uncased'

def lengths2mask(lengths, max_len):
    mask = torch.arange(max_len).to(lengths.device)
    mask = mask.unsqueeze_(0) >= lengths.unsqueeze(1)
    return mask

def _inflate(tensor, times, dim):
    """
    Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
    Args:
        tensor: A :class:`Tensor` to inflate
        times: number of repetitions
        dim: axis for inflation (default=0)
    Returns:
        A :class:`Tensor`
    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> b = ._inflate(a, 2, dim=1)
        >> b
        1   2   1   2
        3   4   3   4
        [torch.LongTensor of size 2x4]
        >> c = _inflate(a, 2, dim=0)
        >> c
        1   2
        3   4
        1   2
        3   4
        [torch.LongTensor of size 4x2]
    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim=2048, output_dim=768, layers=1, pretrained=False):
        super(ImageEncoder, self).__init__()
        self.box_encoding = nn.Sequential(
            nn.Linear(5, 100),
            nn.PReLU(),
            nn.Linear(100, input_dim)
        )
        # self.encoder = nn.MultiheadAttention(input_dim, 8, dropout=0.01)
        self.encoders = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        for i in range(layers):
            self.encoders.append(nn.MultiheadAttention(input_dim, 16, dropout=0.1))
            self.activations.append(nn.PReLU())
            self.ffs.append(nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.PReLU(),
                nn.Linear(4096, 2048),
                nn.PReLU()
            ))
        # self.relu = nn.ReLU(inplace=True)

        self.dense_summary = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            # nn.PReLU(),
            # nn.Linear(256, 128),
            # nn.PReLU(),
            # nn.Linear(128, 1),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim)
        )

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        ckpt = torch.load('/home/dingyuhui/PythonProjects/kdd_cup/checkpoints/image_encoder4.pth',
                          map_location='cpu')
        self.load_state_dict(ckpt, strict=False)

    def forward_multiheadattenton(self, encoder, ff, activation, x, mask):
        identity = x
        x, _ = encoder(x, x, x, key_padding_mask=mask)
        x = identity + x
        x = activation(x)
        x = ff(x)

        return x

    def forward(self, x, boxes, mask):
        '''
        :param x: (batch, n_regions, dim)
        :return:
        '''
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


        # batch, n_regions, dim = x.size()
        return embedding





class ImageEncoder_(nn.Module):
    def __init__(self, input_dim=2048, output_dim=768, pretrained=False):
        super(ImageEncoder_, self).__init__()
        self.box_encoding = nn.Sequential(
            nn.Linear(5, 100),
            nn.PReLU(),
            nn.Linear(100, 5)
        )
        self.encoder = nn.MultiheadAttention(input_dim, 4)
        self.dense_summary = nn.Sequential(
            nn.Linear(input_dim + 5, 200),
            nn.PReLU(),
            nn.Linear(200, 1)
        )

        self.dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.PReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.prelu = nn.PReLU()

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        ckpt = torch.load('/home/dingyuhui/PythonProjects/kdd_cup/checkpoints/image_encoder.pth', map_location='cpu')
        self.load_state_dict(ckpt)

    def get_params(self):
        return [
            {'params': self.parameters(), 'initial_lr': 1e-5, 'weight_decay': 1e-6},
        ]

    def forward(self, x, boxes, mask):
        '''
        :param x: (batch, n_regions, dim)
        :return:
        '''
        # mask = lengths2mask(lengths, x.size(1))
        boxes = self.box_encoding(boxes)
        indentity = x
        x = x.permute(1, 0, 2)
        x, _ = self.encoder(x, x, x, key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        x = indentity + self.prelu(x)


        summary_score = self.dense_summary(torch.cat([x, boxes], -1)).squeeze_(-1)
        summary_score.masked_fill_(mask, -float('inf'))
        summary_score = torch.softmax(summary_score, -1)
        embedding = self.dense(x)
        embedding = torch.einsum('bid,bi->bd', [embedding, summary_score])


        # batch, n_regions, dim = x.size()
        return embedding




class TextGenerator(nn.Module):
    def __init__(self, vocab_size, word_dim=768):
        super(TextGenerator, self).__init__()
        hidden_dim = word_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(word_dim, hidden_dim, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.start_token = 101
        self.end_token = 102

    def get_params(self):
        return [
            {'params': self.parameters(), 'initial_lr': 1e-3, 'weight_decay': 1e-6},
            # {'params': self.W.parameters(), 'initial_lr': 1e-3},
        ]


    def decode(self, img_embedding, W, depth=30):
        batch, dim = img_embedding.size()
        inputs = torch.ones(batch, dtype=torch.long) * self.start_token
        inputs = inputs.to(img_embedding.device)
        h = img_embedding.unsqueeze(0)
        outputs = [inputs]
        for i in range(depth - 1):
            inputs_embedding = F.embedding(inputs, W, padding_idx=0)
            inputs_embedding = inputs_embedding.unsqueeze(1)
            logits, h = self.rnn(inputs_embedding, h)
            logits = logits.squeeze(1)
            logits = self.dense(logits)
            logits = torch.einsum('bd,kd->bk', [logits, W])
            inputs = logits.argmax(1)
            outputs.append(inputs)
        outputs = torch.stack(outputs, 1)
        return outputs

    def beam_search(self, img_embedding, W, depth=15, topk=5):
        n_classes = W.size(0)
        device = img_embedding.device
        batch, dim = img_embedding.size()
        inputs = torch.ones(batch * topk, dtype=torch.long) * self.start_token
        inputs = inputs.to(device)

        accumulate_logprob = torch.zeros((batch * topk, 1)).to(device)
        accumulate_logprob.data.fill_(-float('inf'))
        accumulate_logprob.index_fill_(0, torch.arange(0, topk * batch, topk).to(device), 0.0)

        pos_index = (torch.arange(batch) * topk).view(-1, 1).to(device)
        stored_predecessors = []
        stored_symbols = []
        stored_scores = []

        h = img_embedding.unsqueeze(0)
        h = h.repeat(1, 1, topk).view(1, topk * batch, -1)
        # eos_indices = None
        for i in range(depth):
            inputs_embedding = F.embedding(inputs, W, padding_idx=0)
            inputs_embedding = inputs_embedding.unsqueeze(1)
            logits, h = self.rnn(inputs_embedding, h)
            logits = logits.squeeze(1)
            logits = self.dense(logits)
            logits = torch.einsum('bd,kd->bk', [logits, W])

            logprob = logits - torch.logsumexp(logits, -1, keepdim=True)
            # if eos_indices is not None:
            #     logprob.masked_fill_(eos_indices, 0)
            accumulate_logprob = accumulate_logprob.repeat(1, n_classes)
            accumulate_logprob += logprob
            accumulate_logprob = accumulate_logprob.view(batch, -1)
            top_logprob, top_idx = accumulate_logprob.topk(topk, 1)

            inputs = (top_idx % n_classes).view(batch * topk, 1).squeeze()

            accumulate_logprob = top_logprob.view(batch * topk, 1)
            stored_scores.append(accumulate_logprob.clone())

            eos_indices = inputs.data.eq(self.end_token)
            if eos_indices.nonzero().dim() > 0:
                accumulate_logprob.data.masked_fill_(eos_indices.unsqueeze(-1), -float('inf'))

            predecessors = (top_idx / n_classes + pos_index.expand_as(top_idx)).view(-1)
            h = h.index_select(1, predecessors)
            stored_predecessors.append(predecessors)
            stored_symbols.append(inputs.unsqueeze(-1))
            # inputs = logprob.argmax(1)
        return self._backtrack(stored_predecessors, stored_symbols, stored_scores, batch, topk, depth, pos_index)

    def _backtrack(self, predecessors, symbols, scores, b, k, depth, pos_index):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        # lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        # if lstm:
        #     state_size = nw_hidden[0][0].size()
        #     h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        # else:
        #     h_n = torch.zeros(nw_hidden[0].size())
        # l = [[depth] * k for _ in range(b)]  # Placeholder for lengths of top-k sequences
        # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, k).topk(k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b  # the number of EOS found
        # in the backward loop below for each batch

        t = depth - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(b * k)
        while t >= 0:
            # Re-order the variables with the back pointer
            # current_output = nw_output[t].index_select(0, t_predecessors)
            # if lstm:
            #     current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            # else:
            #     current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.end_token).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = k - (batch_eos_found[b_idx] % k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    # l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            # output.append(current_output)
            # h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(k)
        # for b_idx in range(b):
        #     l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(b * k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [step.index_select(0, re_sorted_idx).view(b, k, -1) for step in reversed(p)]
        # if lstm:
        #     h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, k, ) for h in step]) for step in
        #            reversed(h_t)]
        #     h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, k, self.hidden_dim) for h in h_n])
        # else:
        #     h_t = [step.index_select(1, re_sorted_idx).view(-1, b, k, self.hidden_dim) for step in reversed(h_t)]
        #     h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, k, self.hidden_dim)
        # s = s.data
        p = torch.stack(p, 2).squeeze(-1)
        return p




    def forward(self, idx, embedding, img_embedding, W):
        # lengths = lengths - 1
        # mask = lengths2mask(lengths, embedding.size(1) - 1)
        # embedding = img_embedding.unsqueeze(1).repeat(1, embedding.size(1), 1)
        self.rnn.flatten_parameters()
        logits, _ = self.rnn(embedding[:, :-1, :], img_embedding.unsqueeze(0))
        self.rnn.flatten_parameters()
        logits = self.dense(logits)

        batch, max_len, d = logits.size()

        logits = torch.einsum('bid,kd->bik', [logits, W])
        logits = logits.view(batch * max_len, -1)
        target = idx[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, target, reduction='none')
        mask = target == 0
        loss = loss.masked_fill_(mask, 0)
        loss = loss.view(batch, max_len)


        loss = loss.sum(1)
        loss = loss.mean()
        return loss


class MultiLabelClassifier(nn.Module):
    def __init__(self, large=False, nhead=4):
        super(MultiLabelClassifier, self).__init__()

        if large:
            bert = BertModel.from_pretrained(pretrained_large)
            dim = 1024
        else:
            bert = BertModel.from_pretrained(pretrained_base)
            dim = 768
        self.W = bert.get_input_embeddings().weight
        self.W.requires_grad_(False)
        self.image_encoder = ImageEncoder(2048, (nhead + 1) * dim)
        self.nhead = nhead
        self.dim = dim

        self.text_generator = TextGenerator(bert.get_input_embeddings().num_embeddings, dim)

        self.start_token = 101
        self.end_token = 102
        self.mask_token = 0


    def forward(self, x, boxes, regions, labels, hard_examples=1000):
        '''
        :param x: (batch, regions, dim)
        :param regions: (batch,)
        :param labels: (batch, n_labels)
        :return: (batch, k)
        '''

        W = self.W
        # logits = torch.einsum('brd,kd->brk', [x, W])
        mask = lengths2mask(regions, x.size(1))
        # mask = mask.unsqueeze_(-1)

        # logits.masked_fill_(mask, -float('inf'))
        # logits = logits.max(1)[0]
        x = self.image_encoder(x, boxes, mask)
        batch, _ = x.size()
        x = x.view(batch, self.nhead + 1, self.dim)


        logits = torch.einsum('bid,kd->bik', [x[:, 1:, :].contiguous(), W])
        logits = logits.max(1)[0]
        target = torch.zeros_like(logits)
        target.scatter_(1, labels, 1)

        target[:, self.mask_token] = 0
        target[:, self.start_token] = 0
        target[:, self.end_token] = 0

        loss_pos = -target * F.logsigmoid(logits)


        if self.training:
            loss_pos = loss_pos.sum(1)
            target[:, self.mask_token] = 1
            loss_neg = - (1. - target) * F.logsigmoid(-logits)

            loss_neg, _ = torch.topk(loss_neg, hard_examples, 1)
            loss_neg = loss_neg.sum(1)

            loss = (loss_pos + loss_neg).mean()

            text_embedding = F.embedding(labels, self.W, 0)

            loss_gen = self.text_generator(labels, text_embedding, x[:, 0, :].contiguous(), self.W)


            return logits, loss, loss_gen
        else:
            loss_pos = loss_pos.sum(1)
            _, index = torch.topk(logits, 15, 1)
            sents = self.text_generator.beam_search(x[:, 0, :].contiguous(), self.W)
            return index, loss_pos, sents


if __name__ == '__main__':
    # net = MultiLabelClassifier()

    # x = torch.randn(3, 8, 2048)
    # boxes = torch.randn(3, 8, 5)
    # regions = torch.LongTensor([1,2,8])
    #
    # labels = torch.LongTensor([[1,2,3],[1,2,0],[100,200,0]])
    #
    # out, loss1, loss2 = net(x, boxes, regions, labels)
    #
    # print(out.size())
    #
    # print(loss)

    net = TextGenerator(1000, 768)
    img_embedding = torch.randn(3, 768)
    W = torch.randn(1000, 768)
    print(net.beam_search(img_embedding, W, depth=6, topk=5)[0])
    print(net.beam_search2(img_embedding, W, depth=6, k=5)[0])
    # print(net.decode(img_embedding, W))








