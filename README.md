# Selfish-RNN Sparse Recurrent Neural Networks with Adaptive Connectivity

This repository is the official implementation of Neurips 2020 submission 6137: Selfish-RNN Sparse Recurrent Neural Networks with Adaptive Connectivity

![](Selfish-RNN.png)

## Requirements

The library requires PyTorch v1.0.1 and CUDA v9.0. 
You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 

## Training
> 📋 We provide the training codes of Selfish stacked-LSTM and Selfish RHN. 

To train Selfish stacked-LSTM on PTB dataset with GPU in the paper, run this command:
```
python main.py --sparse --optimizer sgd --model LSTM --cuda --growth random --death magnitude --redistribution none --nonmono 5 --batch_size 20 --bptt 35 --lr 40 --clip 0.25 --seed 5 --emsize 1500 --nhid 1500 --nlayers 2 --death-rate 0.7 --dropout 0.65 --density 0.33 --epochs 100
```

To train Selfish RHN on PTB dataset with GPU in the paper, run this command:
```
python main.py --sparse --optimizer sgd --model RHN --cuda --tied --couple --seed 42 --nlayers 1 --growth random --death magnitude --redistribution none --density 0.472 --death-rate 0.5 --clip 0.25 --lr 15 --epochs 500 --dropout 0.65 --dropouth 0.25 --dropouti 0.65 --dropoute 0.2 --emsize 830 --nhid 830

```
Options:
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --evaluate (str) - pretrained model path (default none)
* --model (str) - type of recurrent net, choose from RHN and LSTM (default LSTM)
* --optimizer (str) - type of optimizers, choose from sgd (Sparse NT-ASGD) and adam (default sgd)

* --growth (str) - regrow mode. Choose from: momentum, random, gradient (default random)
* --death (str) - pruning mode. Choose from: magnitude, SET, threshold (default magnitude)
* --redistribution (str) - redistribution mode. Choose from: momentum, magnitude, nonzeros, or none. (default none)
* --density (float) - density level (default 0.33)
* --death-rate (float) - initial pruning rate (default 0.5)

## Evaluation 

To evaluate the pre-trained Selfish stacked-LSTM model on PTB, run:

```eval
python main.py --sparse --evaluate mymodel.pth --optimizer sgd --model LSTM --cuda --growth random --death magnitude --redistribution none --nonmono 5 --batch_size 20 --bptt 35 --lr 40 --clip 0.25 --seed 5 --emsize 1500 --nhid 1500 --nlayers 2 --death-rate 0.7 --dropout 0.65 --density 0.33 --epochs 100
```

> 📋To evaluate the pre-trained model, you need to replace the mymodel.pth with your model path and all the training hyper-parameters keep the same.

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Selfish stacked-LSTM, RHN and ONLSTM on PTB:]

| Model name            |   Sparsity   | Validation perplexity  | Test perplexity |
| ----------------------|--------------|----------------------- | --------------- |
| Selfish stacked-LSTM  |    0.33      |         73.79          |      71.72      |
| Selfish RHN           |    0.472     |         62.10          |      60.35      |
| Selfish ONLSTM_1000   |    0.450     |       <a href="https://www.codecogs.com/eqnedit.php?latex=58.17\pm0.06" target="_blank"><img src="https://latex.codecogs.com/gif.latex?58.17\pm0.06" title="58.17\pm0.06" /></a>           |      <a href="https://www.codecogs.com/eqnedit.php?latex=56.31\pm0.10" target="_blank"><img src="https://latex.codecogs.com/gif.latex?56.31\pm0.10" title="56.31\pm0.10" /></a>      |
| Selfish ONLSTM_1300   |    0.450     |       <a href="https://www.codecogs.com/eqnedit.php?latex=57.67\pm0.03" target="_blank"><img src="https://latex.codecogs.com/gif.latex?57.67\pm0.03" title="57.67\pm0.03" /></a>           |      <a href="https://www.codecogs.com/eqnedit.php?latex=55.82\pm0.11" target="_blank"><img src="https://latex.codecogs.com/gif.latex?55.82\pm0.11" title="55.82\pm0.11" /></a>      |

### [Selfish AWD-LSTM-MoS on PTB:]

| Model name            |   Sparsity   | Validation perplexity  | Test perplexity |
| ----------------------|--------------|----------------------- | --------------- |
| Selfish AWD-LSTM-MoS  |    0.450     |         65.96          |      63.05      |

## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
