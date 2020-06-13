import os
import argparse
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
parser = argparse.ArgumentParser(description="train")
parser.add_argument('--local_rank', type=int)
parser.add_argument('--devices', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--addr', type=str, default="127.0.0.1")
parser.add_argument('--port', type=str, default="7878")
args = parser.parse_args()
devices = ','.join([str(s) for s in args.devices])
os.environ["CUDA_VISIBLE_DEVICES"] = devices
os.environ["MASTER_PORT"] = args.port
os.environ["MASTER_ADDR"] = args.addr
from torch import nn
from torch import distributed
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from model import TextEncoder, ImageEncoder, ScoreModel, ContrastiveLoss
from utils_backup import Dataset, collate_fn, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from validation import valid
distributed.init_process_group('nccl')
torch.manual_seed(2020)




if __name__ == '__main__':
    local_rank = distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    checkpoints_dir = '/data/data_dyh/kdd_ckpt/ckpt_main/checkpoints4'
    start_epoch = 0
    use_bert = True
    if not os.path.exists(checkpoints_dir) and local_rank == 0:
        os.makedirs(checkpoints_dir)
    kdd_dataset = Dataset(use_bert=use_bert)
    sampler = DistributedSampler(kdd_dataset)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn, batch_size=130, sampler=sampler, num_workers=15)
    nhead = 4
    text_encoder = TextEncoder(kdd_dataset.unknown_token + 1, 1024, 256, use_bert=use_bert).cuda()
    image_encoder = ImageEncoder(input_dim=2048, output_dim=1024, nhead=nhead)
    image_encoder.load_pretrained_weights()
    image_encoder = image_encoder.cuda()
    score_model = ScoreModel(1024, 256).cuda()
    # text_generator = TextGenerator(text_encoder.embed.num_embeddings).cuda()
    # score_model = ScoreModel(30522, 256, num_heads=1).cuda()
    # category_embedding = CategoryEmbedding(256).cuda()


    optimizer = Adam(image_encoder.get_params() + text_encoder.get_params() + score_model.get_params())

    if start_epoch > 0 and local_rank == 0:
        checkpoints = torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(start_epoch)), 'cpu')
        text_encoder.load_state_dict(checkpoints['query'])
        image_encoder.load_state_dict(checkpoints['item'])
        score_model.load_state_dict(checkpoints['score'])
        # text_generator.load_state_dict(checkpoints['generator'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        print("load checkpoints")
    # generator = iterate_minibatches(iters=(30 - start_epoch) * len(loader), batch_size=256, num_workers=8, root_dir='/home/dingyuhui/dataset/kdd-data', use_bert=use_bert)

    scheduler = ExponentialLR(optimizer, 0.95, last_epoch=start_epoch - 1)
    text_encoder = nn.parallel.DistributedDataParallel(text_encoder, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    image_encoder = nn.parallel.DistributedDataParallel(image_encoder, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    score_model = nn.parallel.DistributedDataParallel(score_model, find_unused_parameters=True,
                                                        device_ids=[local_rank], output_device=local_rank)
    contrastive_loss = ContrastiveLoss(0.9, max_violation=True, reduction='mean')
    # contrastive_loss = ExponentialLoss()
    for epoch in range(start_epoch, 30):
        # tbar = tqdm(loader)
        if local_rank == 0:
            tbar = tqdm(loader)
        else:
            tbar = loader
        losses_manual_mining = 0.
        losses_hard_mining = 0.
        # losses_gen = 0.
        # losses3 = 0.
        # losses_classify = 0.
        text_encoder.train()
        image_encoder.train()
        for i, (query, query_len, features, boxes, obj_len,
                query_neg, query_neg_len, features_neg, boxes_neg, obj_neg_len) in enumerate(tbar):
            optimizer.zero_grad()
            batch_size = query.size(0)
            target = torch.ones(batch_size).cuda()
            query_idx = query.cuda()
            query_len = query_len.cuda()
            obj_len = obj_len.cuda()
            obj_neg_len = obj_neg_len.cuda()

            boxes = boxes.cuda()
            boxes_neg = boxes_neg.cuda()
            # category = category.cuda()
            # category_neg = category_neg.cuda()

            query, hidden = text_encoder(query_idx, query_len)
            query_neg, hidden_neg = text_encoder(query_neg.cuda(), query_neg_len.cuda())

            features = image_encoder(features.cuda(), boxes, obj_len)
            features_neg = image_encoder(features_neg.cuda(), boxes_neg, obj_neg_len)

            # category = category_embedding(category)
            # category_neg = category_embedding(category_neg)

            # loss_gen = text_generator(query_idx, query, features, query_len)


            # score_matrix = score_func(query, features, summary, cross=True)
            score_matrix = score_model(query, hidden, query_len, features, cross=True)
            score_positive = score_matrix.diag()

            # score_neg1 = score_func(query, features_neg, summary)
            score_neg1 = score_model(query, hidden, query_len, features_neg)
            # score_neg2 = score_func(query_neg, features, summary_neg)
            score_neg2 = score_model(query_neg, hidden_neg, query_neg_len, features)

            # loss_manual_mining = -F.logsigmoid(score_positive - score_neg1).mean() \
            #                      - F.logsigmoid(score_positive - score_neg2).mean()
            loss_manual_mining = F.margin_ranking_loss(score_positive, score_neg1, target, margin=0.9) \
                   + F.margin_ranking_loss(score_positive, score_neg2, target, margin=0.9)

            loss_hard_mining = contrastive_loss(score_matrix)
            loss = loss_manual_mining + loss_hard_mining # + 0.5 * loss_gen


            loss.backward()
            optimizer.step()

            distributed.all_reduce(loss_manual_mining.data)
            distributed.all_reduce(loss_hard_mining.data)

            losses_hard_mining += loss_hard_mining.data.item() / len(args.devices)
            losses_manual_mining += loss_manual_mining.data.item() / len(args.devices)

            # losses_hard_mining += loss_hard_mining.item()
            # losses_manual_mining += loss_manual_mining.item()
            # losses_gen += loss_gen.item()
            # losses_classify += loss_classify.item()

            # query, query_len, features, boxes, category, obj_len = next(loader2)
            # query = text_encoder(query.cuda(), query_len.cuda())
            # features = image_encoder(features.cuda())
            # score_matrix = score_model(query, features, obj_len.cuda(), cross=True)
            # loss = contrastive_loss(score_matrix)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # losses3 += loss.item()

            if local_rank == 0:
                total = i + 1
                tbar.set_description('epoch: %d, loss manual mining: %.3f, loss hard mining: %.3f'
                                     % (epoch + 1, losses_manual_mining / total, losses_hard_mining / total))


            # tbar.set_description('epoch: %d, loss1: %.3f, loss2: %.3f'
            #                      % (epoch + 1, losses_manual_mining / (i + 1), losses_hard_mining / (i + 1)))
        scheduler.step(epoch)
        if local_rank == 0:
            checkpoints = {
                'query': text_encoder.module.state_dict(),
                'item': image_encoder.module.state_dict(),
                'score': score_model.module.state_dict(),
                # 'category': category_embedding.state_dict(),
                # 'generator': text_generator.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoints, os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch + 1)))
            # score_model.eval()
            # text_encoder.eval()
            with torch.no_grad():
                valid(epoch + 1, checkpoints_dir, use_bert=use_bert)
        distributed.barrier()
