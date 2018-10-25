import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import TestDataset
from utils import gpu_manage, save_image
from models.gen.unet import UNet


def predict(config, args):
    gpu_manage(args)
    dataset = TestDataset(args.test_dir)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    if config.gen_model == 'unet':
        gen = UNet(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=args.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch)
            if args.cuda:
                x = x.cuda()

            out = gen(x)

            h = 1
            w = 4
            c = 3
            p = config.size

            allim = np.zeros((h, w, c, p, p))
            x_ = x.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            in_nir = x_[3]
            out_rgb = np.clip(out_[:3], -1, 1)
            out_cloud = np.clip(out_[3], -1, 1)
            allim[0, 0, :] = np.repeat(in_nir[None, :, :], repeats=3, axis=0) * 127.5 + 127.5
            allim[0, 1, :] = in_rgb * 127.5 + 127.5
            allim[0, 2, :] = out_rgb * 127.5 + 127.5
            allim[0, 3, :] = np.repeat(out_cloud[None, :, :], repeats=3, axis=0) * 127.5 + 127.5
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*p, c))

            save_image(args.out_dir, allim, i, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = AttrMap(config)

    predict(config, args)
