
## Requirements
* Python3 (tested with 3.5.4)
* GDAL (tested with 2.2.2)
* Numpy (tested with 1.14.2)
* Pillow (tested with 5.0.0)
* OpenCV (tested with 3.3.1)
* MulticoreTSNE (tested with 0.0.1.1)
* colorcorrect (tested with 0.7)
* tqdm (tested with 4.15.0)
* PyTorch (tested with 0.4.1)
* TorchVision (tested with 0.2.1)

## Usage
### Convert RGB-NIR GeoTIFF data into cropped RGB and NIR images
`crop_rgb-nir.py` separates RGB-NIR GeoTIFF data into RGB and NIR data and crops the each data except for blackout part.  
If filename option is specified, it is processed for the specified file with single core cpu.  
If not, it is processed for files in the specified input directory with multicore cpu.  
If you specified `--colorcorrect (-cc)` option, cropped images are colorcorrected using `colorcorrect` module.  

```bash
python crop_rgb-nir.py -i <path_to_input_dir> -o <path_to_output_dir> --filename <filename> -s <crop_size> -cc
```
or
```bash
python crop_rgb-nir.py -i <path_to_input_dir> -o <path_to_output_dir> -s <crop_size> -cc
```

### Making a list file of training data
`make_training_datalist.py` makes a list of training data `train_files.pkl` from cropped RGB images.  
`make_training_datalist.py` saves feature vectors of `fc7` layer of pretrained AlexNet as an intermediate result into `filename_feature.pkl`.  
For details, refer to Sec. 3.2 of [our paper](https://arxiv.org/abs/1710.04835).  

```bash
python make_training_datalist.py -i <path_to_input_dir_or_filename_feature.pkl> -o <path_to_output_dir> -n_d <num_of_training_data> -n_g <square_of_num_of_grids>
```

### Synthesis of cloud images
`make_clouds.py` makes synthesized cloud images using perlin noise.  
The size of the cloud can be adjusted by changing `NoiseOffset` in the `PythonCloud/Config.py`.  

```bash
python make_clouds.py -n <num_of_cloud_images> -o <path_to_output_dir>
```

### Visualizing a feature space (optional)
`feature_space_visualizer.py` make an image visualized 2-D feature space from `filename_feature.pkl`.  

```bash
python feature_space_visualizer.py -i <path_to_filename_feature.pkl> -o <path_to_output_file> -g_n <square_of_num_of_grids>
```

## References
* `PythonCloud/` is refered [`Python-Cloud`](https://github.com/SquidDev/Python-Clouds) repository.  We modified the code of this repository for Python3.  

