# Selfish-RNN Sparse Recurrent Neural Networks with Adaptive Connectivity

This repository is the official implementation of Neurips 2020 submission 6137: Selfish-RNN Sparse Recurrent Neural Networks with Adaptive Connectivity

![](Selfish-RNN.png)

## Requirements

The library requires PyTorch v1.0.1 and CUDA v9.0. 
You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 

## Training

To train Selfish stacked-LSTM on PTB dataset on GPU in the paper, run this command:

```
python main.py --sparse --optimizer sgd --model LSTM --cuda --growth random --death magnitude --redistribution none --nonmono 5 --batch_size 20 --bptt 35 --lr 40 --clip 0.25 --seed 5 --emsize 1500 --nhid 1500 --nlayers 2 --death-rate 0.7 --dropout 0.65 --density 0.33 --epochs 100
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
