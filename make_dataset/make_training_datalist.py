import sys
from pathlib import Path
import argparse
import pickle
import random

import numpy as np
from PIL import Image
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import alexnet
from torchvision import transforms

import ipdb

def alexnet_tsne(path_in, cuda):
    model = alexnet(pretrained=True)
    feature_extracter = nn.Sequential(*list(model.classifier.children())[:-2])

    in_image = torch.FloatTensor(1, 3, 224, 224)

    if cuda:
        model = model.cuda()
        feature_extracter = feature_extracter.cuda()
        in_image = in_image.cuda()

    in_image = Variable(in_image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    filenames = []
    features = []
    for filename in tqdm(list(path_in.glob('**/*.png'))[:1000]):
        image = Image.open(filename).convert('RGB')
        image = image.resize((224, 224))
        image = transform(image).unsqueeze(0)
        in_image.data.copy_(image)
        x = model.features(in_image)
        x = x.view(x.size(0), 256 * 6 * 6)
        feature = feature_extracter(x).data.cpu().numpy()[0]

        filenames.append(filename.resolve())
        features.append(feature)
    features = np.asarray(features)

    tsne = TSNE(n_jobs=4)
    tsne_features = tsne.fit_transform(features)

    filename_feature = [filenames, tsne_features]

    filename = 'filename_feature.pkl'
    with open(str(path_out / filename), 'wb') as f:
        pickle.dump(filename_feature, f)
    print('Saved {}'.format(filename))

    return filename_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to the directory where the input file is located')
    parser.add_argument('--output', '-o', required=True, help='Path to the directory to output the files')
    parser.add_argument('--n_data', '-n_d', type=int, default=2000, help='The number of training data extracted from 2d feature space')
    parser.add_argument('--n_grid', '-n_g', type=int, default=32, help='The number of training data extracted from 2d feature space')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    args = parser.parse_args()

    path_in = Path(args.input)
    path_out = Path(args.output)
    if not path_out.exists():
        path_out.mkdir(parents=True)

    if path_in.is_dir():
        filename_feature = alexnet_tsne(path_in, args.cuda)
    elif path_in.is_file():
        with open(str(path_in), 'rb') as f:
            filename_feature = pickle.load(f)

    max_w, min_w = filename_feature[1][:, 0].max(), filename_feature[1][:, 0].min()
    max_h, min_h = filename_feature[1][:, 1].max(), filename_feature[1][:, 1].min()

    n_grid = args.n_grid
    grid_w = (max_w - min_w) / n_grid
    grid_h = (max_h - min_h) / n_grid

    feature_space = [[[] for _ in range(n_grid)] for _ in range(n_grid)]
    for filename, tsne_feature in zip(filename_feature[0], filename_feature[1]):
        for i in range(n_grid):
            for j in range(n_grid):
                if min_w + grid_w * j <= tsne_feature[0] <= min_w + grid_w * (j + 1) and min_h + grid_h * i <= tsne_feature[1] <= min_h + grid_h * (i + 1):
                    feature_space[i][j].append(filename)

    train_files = []
    n_data = args.n_data if args.n_data < len(filename_feature[0]) else len(filename_feature[0])
    while len(train_files) <= n_data:
        for i in range(n_grid):
            for j in range(n_grid):
                if len(feature_space[i][j]) == 0:
                    continue
                filename = random.choice(feature_space[i][j])
                train_files.append(filename)
                feature_space[i][j].remove(filename)
    
    filename = 'train_files.pkl'
    with open(str(path_out / filename), 'wb') as f:
        pickle.dump(train_files, f)
    print('Saved {}'.format(filename))
