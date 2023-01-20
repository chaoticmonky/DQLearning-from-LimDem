# Deep Q-Learning from Limited Demonstrations
This repository includes implementation of [Deep Q-learning from Demonstrations (DQfLD)](https://bit.ly/deepqfld) as developed for the final project of COMPSCI 691NR at University of Massachusetts, Amherst. For benchmarking, a vanilla Deep Q-learning agent, [Deep Q-learning from Demonstrations (DQfD)](https://arxiv.org/abs/1704.03732) and [Soft Q-learning](https://arxiv.org/abs/1702.08165) are also implemented. The Soft Q-learning agent is implemented from [here](https://github.com/gouxiangchen/soft-Q-learning).

## Install dependencies
Using conda:
```bash
conda env create -n conda-env -f /path/to/environment.yml
```
Running this should install everything needed.
This repository uses the following:
- PyTorch v1.13.1
- gym (LunarLander-v2)
- python 3.10.9
- Jupyter


## How to run this project

After dependencies are installed, you can run any of the Jupyter notebooks. To run DQfLD or DQfD, you have to first train the soft Q learning expert.

*  In a second terminal (does not require TF) :

```bash
python3 main.py
```

## Description of experiments
Checkout the entire report [here](https://bit.ly/deepqfld) which contains details of our experiments and our conclusions.

## Authors
- [Shrayani Mondal](https://www.linkedin.com/in/shrayani-mondal-b5a354171/)
- [Yugantar Prakash](https://www.linkedin.com/in/yugantarp/)

## Acknowledgements
Parts of the code are taken from [陈狗翔](https://github.com/gouxiangchen/soft-Q-learning).