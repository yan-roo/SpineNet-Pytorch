# SpineNet-Pytorch
[SpineNet](https://arxiv.org/abs/1912.05027) is a scale-permuted backbone for object detection, proposed by Google Brain in CVPR 2020. This project is a kind of implementation of SpineNet using mmdetection.

It is highly based on the
* [lucifer443/SpineNet-Pytorch](https://github.com/lucifer443/SpineNet-Pytorch)
* The paper [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027)
* [Official TensorFlow Implementation](https://github.com/tensorflow/tpu/tree/master/models/official/detection)

## Models

| Backbone     | Resolution  | mAP  | Params | FLOPs   | mAP in paper | Params in paper | FLOPs in paper |
| ------------ | ----------  | ---- | ------ | ------- | ------------ | --------------- | -------------- |
| SpineNet-49S |   640x640   | 39.1 | 11.15M | 30.04B  | 39.9         | 12.0M           | 33.8B          |
| SpineNet-49  |   640x640   | 42.7 | 28.31M | 83.7B   | 42.8         | 28.5M           | 85.4B          |
| SpineNet-49  |   896x896   | 42.7 | 28.31M | 83.7B   | 42.8         | 28.5M           | 85.4B          |
| SpineNet-96  |  1024x1024  | ——   | 42.74M | 261.35B | 47.1         | 43.0M           | 265.4B         |
| SpineNet-143 |  1280x1280  | ——   | ——     | ——      | 48.1         | 66.9M           | 524.4B         |
| SpineNet-190 |  1280x1280  | ——   | ——     | ——      | ——           | 163.6M          | 1885B          |

**Note**: The parameters and FLOPs are a little different from paper. More information about models can see in [MODEL_DETAILS.md](docs/MODEL_DETAILS.md)

## Installation

### 1. Install mmdetection

   This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.1.0+8732ed9).
   
   Please refer to [INSTALL.md](docs/INSTALL.md) for more information.

   a. Create a conda virtual environment and activate it.
   ```shell
   conda create -n mmlab python=3.7 -y
   conda activate mmlab
   ```

   b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

   ```shell
   conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
   ```
   c. Install mmcv
   
   ```shell
   pip install mmcv==0.4.3
   ```  
   
   d. Clone the mmdetection repository.

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git checkout 8732ed9
   ```

   d. Install build requirements and then install mmdetection.
   (We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

   ```shell
   pip install -r requirements/build.txt
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   pip install -v -e .  # or "python setup.py develop"
   ```

## Getting started
### 1. Copy the codes to mmdetection directory

```shell
git clone https://github.com/yan-roo/SpineNet-Pytorch.git
cp -r mmdet/ mmdetection/
cp -r configs/ mmdetection/
```

### 2. Prepare dataset (COCO)

```shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip ${train/val/test}2017.zip
mv ${train/val/test}2017  ${mmdetection/data/coco/}
```

  The directories should be arranged like this:

     >   mmdetection
     >     ├── mmdet
     >     ├── tools
     >     ├── configs
     >     ├── data
     >     │   ├── coco
     >     │   │   ├── annotations
     >     │   │   ├── train2017
     >     │   │   ├── val2017
     >     │   │   ├── test2017


### 3. Train a model
#### Train with a single GPU
Modify [config/spinenet/model.py Line5](https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/spinenet_49S_B_8gpu.py#L5)
type='SyncBN' -> type='BN'

```shell
CONFIG_FILE=configs/spinenet/spinenet_49S_B_8gpu.py
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

_[optional arguments]:_ --resume_from ${epoch_.pth} / --validate

#### Train with multiple GPUs

```shell
CONFIG_FILE=configs/spinenet/spinenet_49S_B_8gpu.py
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
### 4. Calculate parameters and FLOPs

```shell
python tools/get_flops.py ${CONFIG_FILE} --shape $SIZE $SIZE
```

### 5. Evaluation

   ```shell
   python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out  ${OUTPUT_FILE} --eval bbox
   ```

More usages can reference [GETTING_STARTED.md](docs/GETTING_STARTED.md) or [MMDetection documentation](https://mmdetection.readthedocs.io/).

## Issues & FAQ

   1. ModuleNotFoundError: No module named 'mmcv.cnn.weight_init'

      ```
      pip install mmcv==0.4.3	
      ```

   2. [ImportError: libtorch_cpu.so: cannot open shared object file: No such file or directory](https://github.com/open-mmlab/mmdetection/issues/2627)

      ```
      rm -r build
      python setup.py develop
      ```
