# UTKFace 
Large Scale Face Dataset for probability estimation

## Introduction
To benchmark the performance of neural networks on the task of probability estimation, we build a synthetic dataset based on [UTKFace](https://susanqq.github.io/UTKFace/). The UTKFace dataset consists of over 20,000 face images with annotations of age. We use the age of the $i$th person $z_i$ to assign them a risk of contracting a disease $p_i= \psi(z_i)$ for a fixed function $\psi: \mathbb{N} \rightarrow [0,1]$. Then we simulate whether the person actually contracts the illness by assigning it a label $y_i=1$ with probability $p_i$, and $y_i=0$ otherwise. The probability-estimation task is to estimate the ground-truth probability $p_i$ from the face image $x_i$, which requires learning to discriminate age and map it to the corresponding risk.

## Risk-assignment function
We consider the risk-assignment function $\psi$ in five scenarios, inspired partly by real-world datasets:
- **Uniform**: Equally-spaced, inspired by weather forecasting: $\psi (z) = z / 100$
- **Sigmoid**: Concentrated near two extremes: $\psi (z) =  σ(25(z/100 - 0.29))$
- **Skewed**: Clustered close to zero, inspired by vehicle-collision detection: $\psi (z) = z / 250$
- **Centered**: Clustered in the center, inspired by cancer-survival prediction: $\psi (z) = z / 300 + 0.35$
- **Discrete**: Discretized: $\psi (z) = 0.2\left[{1}_{\{z > 20\}} + {1}_{\{z > 40\}} + {1}_{\{z > 60\}} + {1}_{\{z > 80\}}\right]+0.1$
![](https://i.imgur.com/YIWRtmK.png)

## Datasets

### Images
Download the input face images [here]()
### Labels
Download the corresponding labels for different risk-assignment functions [here]()

The labels of each face image is embedded in the file name, formated as ```labels_ + [Risk-assignment type] + '_' + [split]```, where
- ```[Risk-assignment type]``` is one of *unif*, *sig*, *skew*, *center*, and *discrete*.
- ```[split]``` is one of train, val, test

## Dataloader
If you are using PyTorch, we provide the dataloader [here](https://github.com/jackzhu727/deep-probability-estimation/blob/main/datasets/simulated_face.py).

### Usage
For dataset with uniform risk-assignment labels:
```
data = FaceDataset(root_dir, 'unif', mode='train')
```
