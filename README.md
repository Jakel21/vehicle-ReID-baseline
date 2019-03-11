# vehicle-ReID
## Introduction
Vehicle ReID is a pytorch-based baseline for training and evaluating deep vehicle re-identification models on reid benchmarks.
## Updates
2019.3.11 update the basic baseline code
## Installation
1. cd to your preferred directory and run git clone https://github.com/Jakel21/vehicl-ReID.
2. Install dependencies by pip install -r requirements.txt (if necessary).
## Datasets
+ [veri](https://github.com/VehicleReId/VeRidataset)
+ [vehicleID](https://pkuml.org/resources/pku-vehicleid.html)
The keys to use these datasets are enclosed in the parentheses. See vehiclereid/datasets/__init__.py for details.Both two datasets need to pull request to the supplier.

## Models
+ resnet50
## Losses
+ cross entropy loss
+ triplet loss
+ mmd loss
## Tutorial
### train
Input arguments for the training scripts are unified in [args.py](./args.py).
To train an image-reid model with cross entropy loss, you can do
```
python train-xent-tri.py \
-s veri \    #source dataset for training
-t veri \    # target dataset for test
--height 128 \ # image height
--width 256 \ # image width
--optim amsgrad \ # optimizer
--lr 0.0003 \ # learning rate
--max-epoch 60 \ # maximum epoch to run
--stepsize 20 40 \ # stepsize for learning rate decay
--train-batch-size 64 \
--test-batch-size 100 \
-a resnet50 \ # network architecture
--save-dir log/resnet50-veri \ # where to save the log and models
--gpu-devices 0 \ # gpu device index
```
### test
Use --evaluate to switch to the evaluation mode. In doing so, no model training is performed.
For example you can load pretrained model weights at path_to_model.pth.tar on veri dataset and do evaluation on VehicleID, you can do
```
python train_imgreid_xent.py \
-s veri \ # this does not matter any more
-t vehicleID \ # you can add more datasets here for the test list
--height 128 \
--width 256 \
--test-size 800 \
--test-batch-size 100 \
--evaluate \
-a resnet50 \
--load-weights path_to_model.pth.tar \
--save-dir log/eval-veri-to-vehicleID \
--gpu-devices 0 \
```
