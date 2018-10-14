import argparse
import random
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to the filename_feature.pkl')
    parser.add_argument('--output', '-o', required=True, help='Path to the output the file')
    parser.add_argument('--n_grid', '-g_n', type=int, default=32, help='The number of training data extracted from 2d feature space')
    args = parser.parse_args()

    path_in = Path(args.input)

    with open(path_in, 'rb') as f:
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

    whole_image = Image.new(mode='RGB', size=(64*n_grid, 64*n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            if len(feature_space[i][j]) == 0:
                continue
            filename = random.choice(feature_space[i][j])
            image = Image.open(filename).convert('RGB').resize((64, 64))
            whole_image.paste(image, (64 * j, 64 * i))

    whole_image.save(args.output)
