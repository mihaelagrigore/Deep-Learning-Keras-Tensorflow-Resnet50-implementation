# Implement ResNet in Tensorflow & Keras from scratch and train on GPU

## Objective  
- Implement ResNet from scratch 
- using Tensorflow and Keras
- train on CPU then switch to GPU to compare speed

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

This repo contains one notebook where I'm building a 50-layer ResNet model from scratch using Keras and training it first on CPU (way too slow), then on Kaggle GPU (for a significant improvement in speed).

<b>How to use it:</b>   
Open the Jupyter Notebook in this folder. You can clone it, download it or just read it here. There is also a link at the top of the Notebook which takes you to the same Notebook on Kaggle.

ResNets were introduced by <a href="https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf">He et al. 2016</a> to bypass the problem or <a href="https://www.youtube.com/watch?v=qhXZsFVxGKo">vanishing and exploding gradients in very deep neural networks</a>. 
<br/><br/>
I want to implement this model myself instead of using an existing library because this will give me a deeper understanding of ResNet and because it's a nice opportunity to learn <a href="https://keras.io/">Keras</a>, a popular open source API for neural networks.
<br/><br/> 
For this project I'm using the <a href="https://www.kaggle.com/alessiocorrado99/animals10">10 Animals dataset available on Kaggle</a>.
<br/><br/> 
If you want to jump right to using a ResNet, have a look at <a href='https://keras.io/api/applications/'>Keras' pre-trained models</a>. In this Notebook I will code my ResNet from scratch not out of need, as implementations already exist, but as a valuable learning process.<br/><br/>

This Notebook was ran on Kaggle, where I there is the option to use either the CPU or GPU. You should therefore use it there or on another platform that offers this options (DeepNote, Google Colab etc) to perform the speed test.   
To see the duration of training on a Kaggle CPU, don't activate the GPU accelerator for this Notebook until you reach section 6. When I publish this Notebook, I can either have the GPU activated from the start for the whole Notebook or not at all, I can't have half-half, for demo purposes. So, in order to see the snail speed of section 5, you should manually deactivate and reactivate once you reach section 6 (at which point, you'll have to re-run everything, but will be fast now).  
