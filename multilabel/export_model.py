from multilabel import MultiLabelClassifier
import torch
import os


if __name__ == '__main__':
    net = MultiLabelClassifier(large=True)
    checkpoint_dir = './checkpoints_large'
    out_dir = './checkpoints'
    epoch = 6
    path = os.path.join(checkpoint_dir, 'model-epoch{}.pth'.format(epoch))
    ckpt = torch.load(path, map_location='cpu')
    net.load_state_dict(ckpt['model'])
    torch.save(net.image_encoder.state_dict(), os.path.join(out_dir, 'image_encoder_large.pth'))