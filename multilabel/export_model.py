from multilabel import MultiLabelClassifier
import torch
import os


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="export")
    parser.add_argument('--epoch', type=int, default=6)
    args = parser.parse_args()
    net = MultiLabelClassifier(large=True)
    checkpoint_dir = './checkpoints_large'
    out_dir = '../user_data'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    epoch = args.epoch
    path = os.path.join(checkpoint_dir, 'model-epoch{}.pth'.format(epoch))
    ckpt = torch.load(path, map_location='cpu')
    net.load_state_dict(ckpt['model'])
    torch.save(net.image_encoder.state_dict(), os.path.join(out_dir, 'image_encoder_large.pth'))