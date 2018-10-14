
## Requirements
* GDAL
* tqdm
* Numpy
* Pillow
* OpenCV
* MulticoreTSNE
* PyTorch
* TorchVision
* colorcorrect

## Usage
### Convert RGB-NIR GeoTIFF data into cropped RGB and NIR images
`crop_rgb-nir.py` separates RGB-NIR GeoTIFF data into RGB and NIR data and crops the each data except for blackout part.  
If filename option is specified, it is processed for the specified file with single core cpu.  
If not, it is processed for files in the specified input directory with multicore cpu.  
If you specified `--colorcorrect (-cc)` option, cropped images are colorcorrected using `colorcorrect` module.  

```bash
python crop_rgb-nir.py -i <path to input dir> -o <path to output dir> --filename <filename> -s <size> -cc
```
or
```bash
python crop_rgb-nir.py -i <path to input dir> -o <path to output dir> -s <size> -cc
```

### Making a list file of training data
`make_training_datalist.py` makes a list of training data `train_files.pkl` from cropped RGB images.  
`make_training_datalist.py` saves feature vectors of `fc7` layer of pretrained AlexNet as an intermediate result in `filename_feature.pkl`.  
For details, refer to Sec. 3.2 of [our paper](https://arxiv.org/abs/1710.04835).  

```bash
python make_training_datalist.py -i <path to input dir or filename_feature.pkl> -o <path to output dir> -n_d <num of training data> -n_g <square of num of grids>
```

### Synthesis of cloud images
`make_clouds.py` makes synthesized cloud images using perlin noise.  
The size of the cloud can be adjusted by changing `NoiseOffset` in the `PythonCloud/Config.py`.  

```bash
python make_clouds.py -n <the number of cloud images> -o <path to output dir>
```

### Visualizing a feature space (optional)
`feature_space_visualizer.py` make an image visualized 2-D feature space from `filename_feature.pkl`.  

```bash
python feature_space_visualizer.py -i <path to `filename_feature.pkl`> -o <path to output file> -g_n <square of num of grids>
```

## References
* `PythonCloud/` is refered [`Python-Cloud`](https://github.com/SquidDev/Python-Clouds) repository.  We modified the code of this repository for Python3.

