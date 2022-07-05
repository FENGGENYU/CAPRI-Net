# CAPRI-Net
Early version of released codes for CAPRI-Net(CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly), Please see [project page](https://fenggenyu.github.io/capri.html).

More processed data and pre-trained weights will coming soon. Please leave your questions in issue page.

### Dependencies

Requirements:
- [Pymesh](https://github.com/PyMesh/PyMesh/releases)

Please use environment.yml to install conda environment.

## Datasets and Pre-trained weights

Point sampling methods are adopted from [IM-Net](https://github.com/czq142857/IM-NET) and [If-Net](https://github.com/jchibane/if-net)

Please download processed data from [here](https://drive.google.com/file/d/1fvuTvW5uKIUq3OF9Ybp3mwnjOPvtQRVC/view?usp=sharing) and pre-trained weights from [here](https://drive.google.com/drive/folders/1Mh5ngnlhi1OqNh0DG1KpZhAQKn5dNa7M?usp=sharing).

The config file spec.json needs to placed in the folder of each experiment.

### Usage

Pre-training the network:
```
python train.py -e {experiment_name} -g {gpu_id} -p 0
```

Fine-tuning the network.
For voxel input
```
python fine-tuning.py -e {experiment_name} -g {gpu_id} --test --voxel --start {start_index} --end {end_index}
```
For point cloud input, please change --voxel to --surface, for example:
```
python fine-tuning.py -e {experiment_name} -g {gpu_id} --test --surface --start 0 --end 1
```

Testing for each shape, example is below:
```
python test.py -e {experiment_name} -g {gpu_id} -p 2 -c best_stage2 --test --voxel --start 0 --end 1 
```

## Citation
If you use this code, please cite the following paper.
```
@InProceedings{Yu_2022_CVPR,
    author    = {Yu, Fenggen and Chen, Zhiqin and Li, Manyi and Sanghi, Aditya and Shayani, Hooman and Mahdavi-Amiri, Ali and Zhang, Hao},
    title     = {CAPRI-Net: Learning Compact CAD Shapes With Adaptive Primitive Assembly},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {11768-11778}
}
```

## Code Reference

The framework of this repository is adopted from [DeepSDF](https://github.com/facebookresearch/DeepSDF)
