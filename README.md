# SANet
By Kaige Li, Qichuan Geng, Zhong Zhou. This repository is an official implementation of the paper "Exploring Scale-Aware Features for Real-Time Semantic Segmentation of Street Scenes", which is under review. The full code will be released after review.

## Highlights
<p align="center">
  <img src="figs/city_score.png" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Comparison of inference speed and accuracy for real-time models on test set of Cityscapes.</span> 
</p>

* **Towards Real-time Applications**: SANet could be directly used for the real-time applications, such as autonomous vehicle and medical imaging.
* **A Novel and Efficient Decoder**: a novel Scale-Aware Decoder is introduced to construct features containing scale-specific information for each position via selective context encoding and feature fusion.
* **More Accurate and Faster**: SANet presents 78.1% mIOU with speed of 103.42 FPS on Cityscapes test set and 77.2% mIOU with speed of 198.5 FPS on CamVid test set.

## Demos

A demo of the segmentation performance of our proposed SANets: Predictions of SANet-100 (left) and SANet-50 (right).
<p align="center">
  <img src="figs/video1_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #1</span>
</p>

<p align="center">
  <img src="figs/video2_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #2</span>
</p>

## Metrics

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS (TensorRT) |
|:-:|:-:|:-:|:-:|
| SANet-50 |  73.7 | 72.7 | pending |
| SANet-75 | 77.6  | 76.6 | pending |
| SANet-100 | 79.1 | 78.1 | pending |

| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS (1080Ti) |
|:-:|:-:|:-:|:-:|
| SANet |-| 77.2 | 198.5 |


## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/sanet
source ~/venv/sanet/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```


## Usage

### 0. Prepare the dataset

* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.


### 1. Training

* Download the ImageNet pretrained models and put them into `pretrained_models/imagenet/` dir.
* For example, train the SANet on Cityscapes with batch size of 12 on one GPU:
````bash
python tools/train.py --cfg configs/SANet_cityscapes.yaml
````
* Or train the SANet on Cityscapes using train and val sets simultaneously with batch size of 12 on one GPU:
````bash
python tools/trainval.py --cfg configs/SANet_cityscapes_trainval.yaml
````

### 2. Evaluation

* Download the finetuned models for Cityscapes and CamVid and put them into `pretrained_models/cityscapes/` and `pretrained_models/camvid/` dirs, respectively.
* For example, evaluate the SANet on Cityscapes val set:
````bash
python tools/eval.py --cfg configs/SANet_cityscapes.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/SANet_Cityscapes_test.pt
````
* Or, evaluate the SANet on CamVid test set:
````bash
python tools/eval.py --cfg configs/SANet_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/SANet_Camvid_test.pt
````
* Generate the testing results of SANet on Cityscapes test set:
````bash
python tools/eval.py --cfg configs/SANet_cityscapes_trainval.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/SANet_trainval_Cityscapes_test.pt 
````

### 3. Speed Measurement

* Measure the inference speed of SANet-100 for Cityscapes:
````bash
python models/speed/sanet_speed.py --c 19 --r 1024 2048
````
* Measure the inference speed of SANet for CamVid:
````bash
python models/speed/sanet_speed.py --c 11 --r 720 960
````

### 4. Custom Inputs

* Put all your images in `samples/` and then run the command below using Cityscapes pretrained SANet for image format of .png:
````bash
python tools/custom.py --p '../pretrained_models/cityscapes/SANet_Cityscapes_test.pth' --t '*.png'
````


## TODO
- [ ] Refactor and clean code
- [ ] Organize all codes and upload them
- [ ] Release complete config, network and training files


## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
* [ICNet-pytorch](https://github.com/liminn/ICNet-pytorch)
* [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)
* [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
* [DDRNet](https://github.com/ydhongHIT/DDRNet)

