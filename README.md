# Deep Learning with Tensorflow & Keras: implement ResNet50 from scratch and train on GPU

## Objective  
- Implement ResNet from scratch 
- using Tensorflow and Keras
- train on CPU then switch to GPU to compare speed

If you want to jump right to using a ResNet, have a look at <a href='https://keras.io/api/applications/'>Keras' pre-trained models</a>. In this repo I am implementing a 50-layer ResNet from scratch not out of need, as implementations already exist, but as a learning process.

## Packages used
- python 3.7.9
- tensorflow 2.7.0 (includes keras)
- scikit-learn 0.24.1
- numpy 3.7.9
- pillow 8.2.0
- opencv-python 4.4.0.46

## GPU support
The following NVIDIA software must be installed on your system:
- NVIDIA® GPU drivers —CUDA® 11.2 requires 450.80.02 or higher.
- CUDA® Toolkit —TensorFlow supports CUDA® 11.2 (TensorFlow >= 2.5.0)
- CUPTI ships with the CUDA® Toolkit.
- cuDNN SDK 8.1.0 cuDNN versions).

## Dataset
For this project I'm using the <a href="https://www.kaggle.com/alessiocorrado99/animals10">10 Animals dataset available on Kaggle</a>.

## Implementation

### Resnet50

ResNet is a family of Deep Neural Networks architectures introduced in 2015 <a href="https://arxiv.org/pdf/1512.03385.pdf">He et al.</a>. The original paper discussed 5 different architectures: 18-, 24-, 50-, 101- and 152-layer Neural Networks. I am implementing the 50-layer ResNet or ResNet50. 

ResNets proposed a solution for the exploding/vanishing gradients problem common when building deeper and deeper NNs: taking the output of one layer and to jumping over a few layers and input this deeper into the neural network. This is called a residual block (also, identity block) and the authors illustrate this mechanism in their article like this:

![image](https://user-images.githubusercontent.com/38474985/151539151-8bc8957e-bf7b-4475-9383-1fe97b5523cf.png)

The identity block can be used when the input x has the same dimension (width and height) as the output of the layer where we are feedforwarding x, othersize the addition wouldn't be possible. When this condition is not met, I use a convolution block like in the image below: 

![image](https://user-images.githubusercontent.com/38474985/151539537-b9536443-1cf5-459a-a172-e9a820f0d3b0.png)

The only difference between the identity block and the convolution block is that the second has another convolution layer (plus a batch normalization) on the skip conection path. The convolution layer on the skip connection path has the purpose of resizing x so that its dimension matches the output and thus I can add those two together.

Following the ResNet50 architecture described in <a href="https://arxiv.org/pdf/1512.03385.pdf">He et al. 2015</a>, the architecture I'm implementing in this repo has the structure illustrated below:

![image](https://user-images.githubusercontent.com/38474985/151537573-1b1b8a42-da7e-4bdb-8be1-1d1b34ae2ea9.png)

## GPU versus CPU training

The easiest way to see the diffence in training duration is to open the notebook in this repository, <a href="https://github.com/mihaelagrigore/ResNet-Keras-code-from-scratch-train-on-GPU/blob/main/resnet-keras-code-from-scratch-train-on-gpu.ipynb">resnet-keras-code-from-scratch-train-on-gpu.ipynb</a>, on Kaggle and follow the instructions for ativating GPU contained in the notebook. This is what I did in my case, as I don't have a separate GPU on my laptop.

To set up GPU support on a physical machine, follow <a href='https://www.tensorflow.org/install/gpu'>these instructions</a>.

## Project contents
```
├── config.yaml               - configuration parameters at project level  
├── example_predict.py        - example prediction script using a pretrained model
├── example_train.py          - example script for training the ResNet50 model on a given dataset
├── images              
│   ├── processed             - processed image data, obtained from raw images, ready for feeding into the model during training  
│   ├── raw                   - raw image data  
│   └── test-samples          - test images for model prediction on unsees images  
├── models                    - folder to save trained models   
│   ├── 202201312229          - saved trained model  
├── requirements.txt          - project requirements  
└── src  
    ├── data                  - scripts for data manipulation 
    │   └── make_dataset.py   - preprocess training data from 'raw' folder and outputs into 'processed' 
    ├── models      
    │   ├── predict_model.py  - implements model prediction procedure  
    │   ├── resnet50.py       - contains class ResNet50, the implementation of the 50 layer ResNet model  
    │   ├── train_model.py    - implements model training procedure  
    └── utils  
        └── basic_functions.py  
 ```
 
 To process the data for obtaining squared images of the pre-defined size (as per model architecture definition), run
  ```
  make_dataset.py --dataset 'Animals-10'
  ```
  from the src/data folder
  
  To train a model, run:
   ```
  example_train.py --help
  ```
  to see available parameters:
  ```
  optional arguments:
  -h, --help            show this help message and exit
  --validation_split VALIDATION_SPLIT
                        How much training data to use for validation ? Default value: 0.2
  --batch_size BATCH_SIZE
                        What batch size to use for training. Default value: 32
  --epochs EPOCHS       How many epochs to train for ? Default: 40, with early stopping callback
  --input_size INPUT_SIZE
                        What input size to set for the ResNet50 model architecture ? Default: '(64, 64)'
  --channels CHANNELS   How many channels do training images have ? Assumed RGB images by default. Default: 3
  --log_level LOG_LEVEL
                        How verbose do you want the logging level ? DEBUG: 10, INFO: 20, WARNING: 30
  --fld FLD             Name of the training data folder. Must be placed inside images/processed. Default: Animals-10

  ```
  and select your preferred training options.
  
  To make predictions using a pre-trained model:
   ```
  example_predict.py --help
  ```  
  and choose the desired setting from:

  ```
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Where to load the pretrained model from ? Default: random pick from inside models folder
  --image IMAGE         Which image to classify (full path) ? Default: no default value, will throw error
```
