# EEG-Decision-SincNet

This github is an open-source repository for fitting Decision SincNet. Decision SincNet is an interpretable, shallow neural network that can be used on multi-channel EEG data to fit Drift-Diffusion Model (DDM).


## Cite us
* [IEEE IJCNN Paper](https://ieeexplore.ieee.org/document/9892272)    @INPROCEEDINGS{9892272,
  author={Sun, Qinhua Jenny and Vo, Khuong and Lui, Kitty and Nunez, Michael and Vandekerckhove, Joachim and Srinivasan, Ramesh},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Decision SincNet: Neurocognitive models of decision making that predict cognitive processes from neural signals}, 
  year={2022},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/IJCNN55064.2022.9892272}}


## How to use

1. Install anaconda based on your OS [link from Anaconda](https://docs.anaconda.com/anaconda/install/)  
2. git clone this repository
3. in your terminal ```cd EEG-Decision-SincNet/```
4. create an conda environment using the .yml file. ```conda env create -f environment.yml```  
   This step might take a while. 


## Organization
```ni_model_*.py``` are the main scripts for running specific model. 'ni' stands for Data from Neuroimage paper (Nunez et al., 2019) [link](https://pubmed.ncbi.nlm.nih.gov/31028925/) 

```nn_models_*.py``` are the neural network models   
```layesr_sinc_spatial.py``` has Sinc Convolution Layer (modeified from [Ravanelli & Yoshua Bengio, 2018](https://arxiv.org/abs/1808.00158) and Separable Convolution Layer.
