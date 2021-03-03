# Lane Detection Using Vanishing Points


## Overview

***

This repository provides the code for our CS294-167 final project: Lane Detection using Vanishing Points. Through the course of this experiment, we primarily used Google Colab. Files for our code is also available on this repository.

Our multi-task CNN which also predicts the scene's vanishing point can be found [here](https://colab.research.google.com/drive/1W4dX96ZmzpDaOq_KsI_l5TSfvjtIG0Fd?usp=sharing).

For reference we have also implemented a naive one-task network which directly outputs semantic segmentation masks of the lane marking (without explicitly finding the vanishing point), code for that network is available [here](https://colab.research.google.com/drive/1g2PUqlCkE_qzPQWWJgNnLV2bGf2KLAu_?usp=sharing).

## Dependencies

***

This codebase requires the following packages:
-PyTorch
-Numpy
-OpenCV2
-SciPy
-Pandas
