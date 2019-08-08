# Tiger Pose Estimation - Ensemble learning on HRNet
An ensemble technique applied on High Resolution net to improve accuracy on pose detection task.
This project was carried out as a part of CVWC-2019 for ICCV 2019 - Amur Tiger re-Identification challenge.

#### Challenge overview - https://cvwc2019.github.io/challenge.html

## Method Description
The key aspect of our approach was to improve on the corner cases on already effective HRNet network. For this reason, the following methodology was adopted:
1. We conducted experiments with multi resolution images to test the effect of resolution on      the model. We finally settled on 640x480 input size. 
2. During training we adopted a 5 fold split on the entire train+validation dataset. 
3. For improving the accuracy during inference, the 5 fold split was ensembled using multiple      approaches - average ensemble, bagging  ensemble and random forest ensemble for obtaining      the best results from the solution. For the submission we have selected average ensemble as    it performed the best in our experiments.
4. All the models used we trained on HRNet-W32 network which were pre-trained on ImageNet          dataset.

### Folder structure for pretrained model
```diff
HEAD
|__models
   |__pytorch
      |__imagenet
         |__hrnet_w32-36af842e.pth
```
### Folder structure for dataset
```diff
HEAD
|__data
   |__tiger
      |__images
      |  |__train
      |  |__test
      |  |__val
      |__annotations
         |__<train_annotations>
         |__<test_annotations>
         |__bb_predictions_pose_test.json <GT bboxes for test dataset>
```
### Folder structure for output directory (Taken care by the code)
```diff
HEAD
|__output (or whatever name was specified in the config or Notebook)
   |__<dataset>
       |__pose_hrnet
          |__<config>
             |__contains results and intermediate training results
```
## Data Preparation
All trained models and pretrained models are available here:
[Drive_Link_For_models](https://drive.google.com/drive/folders/1tWkShwPoSZUsJlx3ijDv_h-P2PI6I5YW?usp=sharing)

The pretrained model needs to be placed as per the instructions above for the pretrained model.

For the trained models, each model must be placed in separate directories as the code looks for model directories and not model-paths( we are working on fixing that as well)

for eg:
```diff
{ROOT}
|__trained
   |__<resolution>
       |__model1
       |  |__final_state.pth
       |__model2
       |  |__final_state.pth
       |....
       :
       :
```

## Running the Experiments
The experiment requires you to create 5 directories named output1, output2.... These will store the trained output models.
The trained models are fed to a predictor which runs evaluation on the val/test dataset to obtain the final outputs.
The outputs and the GTs are passed to the evaluator script which scores and returns the performance metrics to us.
To run the above mentioned scenario do the following:
1. Create your conda environment from the '.yaml' file provided in the root directory.
```diff
conda env create -f pose-env.yaml
```
2. Run the commands
```diff
   mkdir output
   mkdir log
 ```
2. Go to the ```lib ``` directory and run ``` make ```. This builds the nms library
3. Also install pycocotools 
4. Your basic setup is ready.

### Training the 5-Fold training model
Run the following command:
```diff
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
 ```
 Remember to change the config file path to suite your requirements.
 
 ### Test a sample
 ```diff
 python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
  ```
  ### Ensemble Testing
  All the ensemble code has been kept in the interactive python Notebook situated here:
  ```diff
  <root>/tools/ensemble-hrnet.ipynb
  ```
  Follow the instructions in sequence in the notebook.
## References
[1] Deep High-Resolution Representation Learning for Human Pose Estimation. Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR 2019. [download](https://arxiv.org/pdf/1902.09212.pdf)
