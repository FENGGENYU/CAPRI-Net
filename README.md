# CAPRI-Net
Early version of released codes for CAPRI-Net, Please see [project page](https://fenggenyu.github.io/capri.html).

More processed data and pre-trained weights will coming soon. Please leave your questions in issue page.

### Dependencies

Requirements:
- [Pymesh](https://github.com/PyMesh/PyMesh/releases)

Please use environment.yml to install conda environment.

## Datasets and Pre-trained weights

Point sampling methods are adopted from [IM-Net](https://github.com/czq142857/IM-NET) and [If-Net](https://github.com/jchibane/if-net)

Please download processed data from [here](https://drive.google.com/file/d/1fvuTvW5uKIUq3OF9Ybp3mwnjOPvtQRVC/view?usp=sharing) and pre-trained weights from [here](https://drive.google.com/drive/folders/1Mh5ngnlhi1OqNh0DG1KpZhAQKn5dNa7M?usp=sharing).

### Usage

Pre-training the network:
```
python train.py -e {experiment_name} -g {gpu_id} -p 0
```

Fine-tuning the network:
For voxel input
```
python fine-tuning.py -e {experiment_name} -g {gpu_id} --test --voxel --start 0 --end 1
```
For point cloud input, please change --voxel to --surface
```
python fine-tuning.py -e {experiment_name} -g {gpu_id} --test --surface --start 0 --end 1
```

Testing for each shape:
```
python test.py -e {experiment_name} -g {gpu_id} -p 2 -c best_stage2 --test --voxel --start 0 --end 1 
```

## Citation
If you use this code, please cite the following paper.
```
@article{Capri_Yu,
author = {Fenggen Yu and Zhiqin Chen and Manyi Li and Aditya Sanghi and Hooman Shayani and Ali Mahdavi{-}Amiri and Hao Zhang},
title = {CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly},
year = {2021},
url = {https://arxiv.org/abs/2104.05652},
biburl = {https://dblp.org/rec/journals/corr/abs-2104-05652.bib},
}
```