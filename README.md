# ML-STGNet
PyTorch implementation of "Multilevel Spatial–Temporal Excited Graph Network for Skeleton-Based Action Recognition", TIP.
[[PDF](https://ieeexplore.ieee.org/document/9997556/)]

## Data Preparation
Four datasets are used in our experiments.

### NTU-60 and NTU-120 
1. Request datset here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
2. Download the skeleton-only datasets:
  - `nturgbd_skeletons_s001_to_s017.zip`  (NTU RGB+D 60)
  - `nturgbd_skeletons_s018_to_s032.zip`  (NTU RGB+D 120, on top of NTU RGB+D 60)

### Skeleton-Kinetics
1. Download dataset from ST-GCN repo: https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton

### Toyota Smarthome
1. Download the raw data from https://github.com/YangDi666/SSTA-PRS#refined-pose-data (skeleton-v2.0 refined by SSTA-PRS)

#### Data Structure
Put downloaded data into the following directory structure:
```
- data/
  - kinetics_raw/
    - kinetics_train/
      ...
    - kinetics_val/
      ...
    - kinetics_train_label.json
    - keintics_val_label.json
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
    - NTU_RGBD_samples_with_missing_skeletons.txt
    - NTU_RGBD120_samples_with_missing_skeletons.txt
   - smarthone_raw/
    - smarthone_skeletons/
      ...    
```
