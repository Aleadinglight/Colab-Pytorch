# Pytorch in Google Colab

Implement deep learning with Pytorch in Google Colab 

## About Google Colab
Read more here: https://colab.research.google.com/notebooks/welcome.ipynb

## About Pytorch
Read more here: https://pytorch.org/

Pytorch has currently released a new version, read more at: https://pytorch.org/2018/04/22/0_4_0-migration-guide.html

## Implementation

1. [Install Pytorch:](../master/Colab_With_Pytorch.ipynb) Follow the instruction here. After this, the library is already installed in your Google Drive location. You don't have to reinstall it next time.

2. [Pytorch Tensors:](../master/PytorchTensorsWithGraph.py) Example on how to use PyTorch Tensors to fit a two-layer network to random data.

<img src="../master/picture/2.png" width="300">

3. [Pytorch Autograd:](../master/GradWithGraph.py) Implement Autograd on the previous problem.

<img src="../master/picture/3.png" width="300">

4. [Recurrent Neural Network:](../master/rnn.ipynb) Using recurrent neural network to predict a sequence: 'hihell' -> 'hihello'. In this case, I used seq to seq model. That is, each letter will predict the next letter: 'hihell' -> 'ihello'.

&nbsp;&nbsp;&nbsp;&nbsp;The model looks like this, assuming that batch_size = 1

&nbsp;&nbsp;&nbsp;&nbsp;<img src="../master/picture/rnn.jpg" width="300">

&nbsp;&nbsp;&nbsp;&nbsp;The traning result:

&nbsp;&nbsp;&nbsp;&nbsp;<img src="../master/picture/4.png" width="300">

5. [Recurrent Neural Network for Classification:](../master/RnnClassification.ipynb) Using RNN to classify. Note: When using embedding table, remember that the vocab_size of Embedding() must be bigger than the biggest element in the input tensor.

&nbsp;&nbsp;&nbsp;&nbsp;<img src="../master/picture/rnnClassification.jpg" width="300">
