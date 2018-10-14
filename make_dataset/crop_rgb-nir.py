import argparse
from pathlib import Path

import gdal
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from colorcorrect import algorithm as cca


def crop(filename, input_dir, output_dir, size=256, colorcorrect=False):
    filename = Path(filename)
    name = filename.name.replace(filename.suffix, '')

    path_in = Path(input_dir) / filename
    path_out_rgb_dir = Path(output_dir) / 'RGB'
    path_out_nir_dir = Path(output_dir) / 'NIR'

    ds = gdal.Open(str(path_in))
    data_matrix = ds.ReadAsArray()

    rgb = np.transpose(data_matrix[:3, :, :], (1, 2, 0))
    nir = data_matrix[3, :, :]

    height, width = nir.shape
    hnum = height // size
    wnum = width // size

    num = 0
    rejected = 0
    for h in range(hnum):
        for w in range(wnum):
            crop_rgb = rgb[size*h:size*h+size, size*w:size*w+size]
            crop_nir = nir[size*h:size*h+size, size*w:size*w+size]

            if not check_nodata(crop_rgb, crop_nir):
                rejected += 1
                continue
            
            if colorcorrect:
                crop_rgb = cca.stretch(cca.grey_world(crop_rgb))
            
            path_out_rgb = path_out_rgb_dir / '{}_{:06d}.png'.format(name, num)
            path_out_nir = path_out_nir_dir / '{}_{:06d}.png'.format(name, num)
            num += 1

            Image.fromarray(crop_rgb).save(path_out_rgb)
            Image.fromarray(crop_nir).save(path_out_nir)

    print('{} images rejected'.format(rejected))
    print('{} images generated'.format(num))


def check_nodata(rgb, nir):
    height, width = nir.shape

    for h in range(height):
        for w in range(width):
            if rgb[h, w, 0] == 0 and rgb[h, w, 1] == 0 and rgb[h, w, 2] == 0 and nir[h, w] == 0:
                return False
    return True


def multi_process(input_dir, output_dir, size, colorcorrect):
    tiffiles = Path(input_dir).glob('*.tif')
    Parallel(n_jobs=-1)([delayed(crop)(filename, input_dir, output_dir, size, colorcorrect) for filename in tiffiles])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to the directory where the input file is located')
    parser.add_argument('--output', '-o', required=True, help='Path to the directory to output the files')
    parser.add_argument('--filename', default=None, help='filename *.tif')
    parser.add_argument('--size', '-s', type=int, default=256, help='The size of cropped image')
    parser.add_argument('--colorcorrect', '-cc', action='store_true')
    args = parser.parse_args()

    path_out_rgb_dir = Path(args.output) / 'RGB'
    path_out_nir_dir = Path(args.output) / 'NIR'

    if not path_out_rgb_dir.exists():
        path_out_rgb_dir.mkdir(parents=True)
        print('{} was made'.format(path_out_rgb_dir))

    if not path_out_nir_dir.exists():
        path_out_nir_dir.mkdir(parents=True)
        print('{} was made'.format(path_out_nir_dir))

    if args.filename is not None:
        crop(args.filename, args.input, args.output, args.size, args.colorcorrect)
    else:
        multi_process(args.input, args.output, args.size, args.colorcorrect)
