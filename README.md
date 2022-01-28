# Implement ResNet in Tensorflow & Keras from scratch and train on GPU

## Objective  
- Implement ResNet from scratch 
- using Tensorflow and Keras
- train on CPU then switch to GPU to compare speed

If you want to jump right to using a ResNet, have a look at <a href='https://keras.io/api/applications/'>Keras' pre-trained models</a>. In this Notebook I will code my ResNet from scratch not out of need, as implementations already exist, but as a valuable learning process.

## Packages used
- tensorflow 2.4.1 (includes keras too)
- python 3.7.9
- scikit-learn 0.24.1
- numpy 3.7.9

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

Following the ResNet50 architecture described in <a href="https://arxiv.org/pdf/1512.03385.pdf">He et al. 2015</a>, the architecture I'm implementing in this repo has the structure illustrated below:

![image](https://user-images.githubusercontent.com/38474985/151537573-1b1b8a42-da7e-4bdb-8be1-1d1b34ae2ea9.png)

## GPU versus CPU training

The easiest way to see the diffence in training duration is to open the notebook in this repository, <a href="https://github.com/mihaelagrigore/ResNet-Keras-code-from-scratch-train-on-GPU/blob/main/resnet-keras-code-from-scratch-train-on-gpu.ipynb">resnet-keras-code-from-scratch-train-on-gpu.ipynb</a>, on Kaggle and follow the instructions for ativating GPU contained in the notebook. This is what I did in my case, as I don't have a separate GPU on my laptop.

To set up GPU support on a physical machine, follow <a href='https://www.tensorflow.org/install/gpu'>these instructions</a>.
