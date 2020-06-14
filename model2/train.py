import os
import argparse
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
parser = argparse.ArgumentParser(description="train")
parser.add_argument('--local_rank', type=int)
parser.add_argument('--devices', type=int, nargs='+', default=[0,1])
parser.add_argument('--addr', type=str, default="127.0.0.1")
parser.add_argument('--port', type=str, default="7718")
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

from model import ScoreModel, ImageEncoder
# from image_encoders import ImageEncoder
from utils import Dataset, collate_fn, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from validation import valid
distributed.init_process_group('nccl')
torch.manual_seed(2020)
# torch.autograd.set_detect_anomaly(True)



if __name__ == '__main__':
    local_rank = distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    checkpoints_dir = './checkpoints2'
    start_epoch = 0
    use_bert = True
    if not os.path.exists(checkpoints_dir) and local_rank == 0:
        os.makedirs(checkpoints_dir)
    kdd_dataset = Dataset(use_bert=use_bert)
    sampler = DistributedSampler(kdd_dataset)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn, batch_size=150, sampler=sampler, num_workers=20)
    nhead = 4
    score_model = ScoreModel(kdd_dataset.unknown_token + 1, 1024, 1024, use_bert=use_bert).cuda()
    image_encoder = ImageEncoder(input_dim=2048, output_dim=1024, nhead=nhead)
    image_encoder.load_pretrained_weights()
    image_encoder = image_encoder.cuda()
    # text_generator = TextGenerator(score_model.embed.num_embeddings).cuda()
    # score_model = ScoreModel(30522, 256, num_heads=1).cuda()
    # category_embedding = CategoryEmbedding(256).cuda()


    optimizer = Adam(score_model.get_params() + image_encoder.get_params())

    if start_epoch > 0 and local_rank == 0:
        checkpoints = torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(start_epoch)))
        score_model.load_state_dict(checkpoints['score'])
        image_encoder.load_state_dict(checkpoints['item'])
        # text_generator.load_state_dict(checkpoints['generator'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        print("load checkpoints")
    # generator = iterate_minibatches(iters=(30 - start_epoch) * len(loader), batch_size=256, num_workers=8, root_dir='/home/dingyuhui/dataset/kdd-data', use_bert=use_bert)

    scheduler = ExponentialLR(optimizer, 0.95, last_epoch=start_epoch - 1)
    score_model = nn.parallel.DistributedDataParallel(score_model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    # image_encoder = nn.parallel.DistributedDataParallel(image_encoder, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    # contrastive_loss = ContrastiveLoss(0.9, max_violation=True, reduction='mean')
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
        score_model.train()
        image_encoder.train()
        for i, (query, query_len, features, boxes, obj_len,
                query_neg, query_neg_len, features_neg, boxes_neg, obj_neg_len) in enumerate(tbar):
            optimizer.zero_grad()
            batch_size = query.size(0)
            target = torch.ones(batch_size).cuda()
            query = query.cuda()
            query_len = query_len.cuda()
            obj_len = obj_len.cuda()
            obj_neg_len = obj_neg_len.cuda()

            boxes = boxes.cuda()
            boxes_neg = boxes_neg.cuda()
            # category = category.cuda()
            # category_neg = category_neg.cuda()



            features = image_encoder(features.cuda(), boxes, obj_len)
            features_neg = image_encoder(features_neg.cuda(), boxes_neg, obj_neg_len)
            score_positive = score_model(query, query_len, features)

            query_neg, query_neg_len = query_neg.cuda(), query_neg_len.cuda()

            score_neg1 = score_model(query_neg, query_neg_len, features)
            score_neg2 = score_model(query, query_len, features_neg)


            score = torch.cat([score_positive, score_neg1, score_neg2])
            target = torch.cat([target, 1 - target, 1 - target])
            loss_manual_mining = F.binary_cross_entropy_with_logits(score, target)
            loss = loss_manual_mining  # + loss_hard_mining # + 0.5 * loss_gen


            loss.backward()
            optimizer.step()

            distributed.all_reduce(loss_manual_mining.data)
            losses_manual_mining += loss_manual_mining.data.item() / len(args.devices)


            if local_rank == 0:
                total = i + 1
                tbar.set_description('epoch: %d, loss manual mining: %.3f'
                                     % (epoch + 1, losses_manual_mining / total))


            # tbar.set_description('epoch: %d, loss1: %.3f, loss2: %.3f'
            #                      % (epoch + 1, losses_manual_mining / (i + 1), losses_hard_mining / (i + 1)))
        scheduler.step(epoch)
        if local_rank == 0:
            checkpoints = {
                'score': score_model.module.state_dict(),
                'item': image_encoder.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoints, os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch + 1)))
            # score_model.eval()
            # score_model.eval()
            with torch.no_grad():
                valid(epoch + 1, checkpoints_dir, use_bert=use_bert)
        distributed.barrier()
