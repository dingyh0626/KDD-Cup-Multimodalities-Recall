import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

from multilabel import MultiLabelClassifier
from utils import Dataset, collate_fn, DataLoader
from tqdm import tqdm

from validation import valid





if __name__ == '__main__':
    # checkpoints_dir = './checkpoints3'
    large = True
    checkpoints_dir = './checkpoints_large'
    start_epoch = 0
    use_bert = True
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    kdd_dataset = Dataset(use_bert=use_bert)
    loader = DataLoader(kdd_dataset, collate_fn=collate_fn, batch_size=200, shuffle=True, num_workers=20)

    model = MultiLabelClassifier(large=large).cuda()
    # model = nn.DataParallel(model)
    params = model.parameters()
    optimizer = Adam(params, lr=1e-4, weight_decay=1e-6)


    if start_epoch > 0:
        checkpoints = torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(start_epoch)))
        model.load_state_dict(checkpoints['model'])
        # score_model.load_state_dict(checkpoints['score'])
        # item_embedding.load_state_dict(checkpoints['item'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        print("load checkpoints")
    # model = torch.nn.DataParallel(model)
    scheduler = ExponentialLR(optimizer, 0.96, last_epoch=start_epoch - 1)
    for epoch in range(start_epoch, 20):
        tbar = tqdm(loader)
        losses_clf = 0.
        losses_gen = 0.
        model.train()
        for i, (query, query_len, features, boxes, obj_len) in enumerate(tbar):
            query = query.cuda()
            features = features.cuda()
            obj_len = obj_len.cuda()
            boxes = boxes.cuda()
            optimizer.zero_grad()
            _, loss_clf, loss_gen = model(features, boxes, obj_len, query)
            loss_clf = loss_clf.mean()
            loss_gen = loss_gen.mean()
            loss = loss_clf + loss_gen
            loss.backward()
            optimizer.step()
            losses_clf += loss_clf.item()

            losses_gen += loss_gen.item()

            tbar.set_description('epoch: %d, loss clf: %.3f, loss gen: %.3f'
                                 % (epoch + 1, losses_clf / (i + 1), losses_gen / (i + 1)))
        scheduler.step(epoch)

        checkpoints = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoints, os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch + 1)))
        valid(epoch + 1, checkpoints_dir, use_bert=True, large=large)
        # score_model.eval()
        # query_embedding.eval()
        # with torch.no_grad():
        #     valid(epoch + 1, checkpoints_dir, version=2, use_bert=use_bert)

