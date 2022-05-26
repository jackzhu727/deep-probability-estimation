## Deep Probability Estimation

This website contains results, code, and pre-trained models from [Deep Probability Estimation](https://arxiv.org/abs/2111.10734) by Sheng Liu\*, Aakash Kaku\*, Weicheng Zhu\*, Matan Leibovich\*,  Sreyas Mohan\*, Boyang Yu, Laure Zanna, Narges Razavian, Carlos Fernandez-Granda [\* - Equal Contribution].

## What Is Probability Estimation?
Estimating probabilities reliably is of crucial importance in many real-world applications such as weather forecasting, medical prognosis, or collision avoidance in autonomous vehicles. This work investigates how to use deep neural networks to estimate probabilities from high-dimensional data such as climatological radar maps, histopathology images, and dashcam videos. 


<!-- ![website_image_2](https://user-images.githubusercontent.com/32464452/158398042-b6d0d993-3ea7-4a24-859f-bb45a00dec52.png) -->


<!-- ![website_image_2](https://github.com/jackzhu727/deep-probability-estimation/blob/main/docs/figs/fig1.png) -->
<img align="center" width="720" src="https://user-images.githubusercontent.com/32464452/158396497-98d4ac2a-8668-4700-8ff3-e3ec6611e892.png">

Probability-estimation models are trained on observed outcomes (<img src="https://latex.codecogs.com/gif.latex?y_i" />) (e.g. whether it has rained or not, or whether a patient has died or not), because the ground-truth probabilities (<img src="https://latex.codecogs.com/gif.latex?p_i" />) of the events of interest are typically unknown. The problem is therefore analogous to binary classification, with the important difference that the main objective at inference is to estimate probabilities (<img src="https://latex.codecogs.com/gif.latex?\hat{p}" />) rather than predicting the specific outcome.


## Early Learning and Memorization in Probability Estimation
Prediction models based on deep learning are typically trained by minimizing the cross entropy between the model output and the training labels. This cost function is  guaranteed to be well calibrated in an infinite-data regime, as illustrated by the figure below (1st column). Unfortunately, in practice, prediction models are trained on finite data. In this case, we observe that neural networks indeed eventually overfit and *memorize* the observed outcomes completely. Moreover, the estimated probabilities collapse to 0 or 1 (2nd column). However, calibration is preserved during the first stages of training (3rd column), which we call *early learning*. In our paper we provide a theoretical analysis showing that this is a general phenomenon that occurs even for linear models the dimension of the input data is large (Theorem 4.1 in the paper). Our proposed method exploits the early-learning phenomenon to obtain an improved model that is still well calibrated (4th column).

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144642950-e477d168-793a-4d9e-818a-5e4c65b637c6.png" alt>
</p>

## Proposed Method: Calibrated Probability Estimation (CAPE)
We propose Calibrated Probability Estimation (CaPE). Our starting point is a model obtained via early stopping using validation data on the cross-entropy loss. CaPE is designed to produce a discriminative model that is well calibrated. This is achieved by alternatively minimizing two loss functions: (1) a *discrimination loss* dependent on the observed binary outcomes, and (2) a *calibration loss*, which ensures that the output probabilities remain calibrated. 
    
The following figures shows the learning curves of cross-entropy (CE) minimization and CaPE, smoothed with a 5-epoch moving average. After an early-learning stage where both training and validation losses decrease, CE minimization overfits (1st and 2nd column), with disastrous consequences in terms of probability estimation (3rd and 4th column, which show the mean squared error and Kullback Leibler divergence with respect to ground-truth probabilities). In contrast, CaPE prevents overfitting, continuing to improve the model, while maintaining calibration.

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144643659-6537f6eb-ee52-46f5-ba0e-86e42dd90208.png" alt>
</p>


<!-- <p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144642950-e477d168-793a-4d9e-818a-5e4c65b637c6.png" alt>
  <em> <br /> When trained on infinite data (i.e. resampling outcome labels at each epoch according to ground-truth probabilities), models minimizing cross-entropy are well-calibrated (first column). The top row shows results for the synthetic Discrete scenario (top). The bottom row shows results for the Linear scenario (dashed line indicates perfect calibration). However, when trained on fixed observed outcomes, the model eventually overfits, and the probabilities collapse to either 0 or 1 (second column). This is mitigated via early stopping (i.e. selecting the model based on validation cross-entropy loss), which yields relatively good calibration (third column). The proposed Calibration Probability Estimation (CaPE) method exploits this to further improve the model discrimination while ensuring that the output remains well-calibrated.</em>
</p> -->


## Synthetic dataset - Face-Based Risk Prediction
To benchmark probability-estimation methods, we built a synthetic dataset based on UTKFace (Zhang et al., 2017b), containing face images and associated ages. We use the age of the person to assign them a probability of contracting a disease. Then we simulate whether the person actually contracts the illness or not with the assigned probability. We use different functions to map from age to probabilities in order to simulate different realistic scenarios. More detais are available [here](https://github.com/jackzhu727/deep-probability-estimation/tree/main/examples/UTKFace).
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/158399694-386ff3ec-6464-4e0f-952f-21c954a953a9.PNG" alt>
  <em> <br /> Examples from Face-based risk prediction dataset (Linear scenario: The function used to convert age to a probability is a linear function).</em>
</p>
 
We use the benchmark dataset to compare our proposed approach with existing methods, showing that it outperforms them across different scenarios.

  <p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144645100-8beb337d-3457-46c5-acd7-b8f88b849b1c.png" alt>
</p>

## Evaluation metrics 
Probability estimation shares similar target labels and network outputs with binary classification. However, classification accuracy is __not__ an appropriate metric for evaluating probability-estimation models due to the inherent uncertainty of the outcomes. 

For our synthetic dataset, we have access to the ground-truth probability labels and can use them to evaluate performance. A reasonable metric in this case is the mean squared error (<img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" />) between the estimated probability and the ground truth probability.

In practice, *ground-truth probabilities are not available*. In that case, traditional forecasting metrics such as Brier score, calibration metrics like ECE, MCE, KS-error, or classification metrics AUC that can be used to evaluate the performance of the model. To determine what metric is more appropriate, we use the synthetic dataset to compare different metrics to the __gold-standard__  <img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" /> that uses ground-truth probabilities. Brier score is found to be highly correlated with <img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" />, in contrast to the classification metric AUC and the calibration metrics ECE, MCE and KS-Error.
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144640753-700c8858-09f1-4503-971f-aa73b0918c14.png" />
</p>

## Real-world datasets

We evaluate the proposed method on three probability estimation tasks using real-world data.

- **Survival of Cancer Patients**: Based on the Hematoxylin and Eosin slides of non-small cell lung cancers from The Cancer Genome Atlas Program (TCGA), we estimate the 5-year survival probability of cancer patients. See [here](https://github.com/jackzhu727/deep-probability-estimation/tree/main/examples/cancer_survival) for more details.

- **Weather Forecasting**: We use the German Weather service dataset, which contains quality-controlled rainfall-depth composites from 17 operational Doppler radars. We use 30 minutes of precipitation data to predict if the mean precipitation over the area covered will increase or decrease one hour after the most recent measurement. Three precipitation maps from the past 30 minutes serve as an input. See [here](https://github.com/jackzhu727/deep-probability-estimation/tree/main/examples/Weather_preidction) for details.

- **Collision Prediction**: We use 0.3 seconds of real dashcam videos from the __YouTubeCrash__ dataset as input, and predict the probability of a collision in the next 2 seconds.

On all the three real-world datasets, CaPE outperforms the existing calibration approaches (when compared using the Brier score which was found to capture the probability estimation performance in the absence of the ground truth probabilities)

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144646458-3b68b90d-0cca-46b7-89ab-ba5dfea4584c.png" alt>
</p>

In addition, the following reliability diagrams show that CaPE produces well calibrated probabilities for the three real-world datasets.

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144646569-53fb0e4b-9a14-45e2-a6f7-d6a203dcd89a.png" alt>

</p>

## Video presentation
[![video_presentation](https://user-images.githubusercontent.com/32464452/158455015-1af15b7e-136c-4ce5-96b7-2ad92c986b3f.PNG)](https://youtu.be/cM7I357nrpA?t=3701)

[Slides](https://github.com/jackzhu727/deep-probability-estimation/blob/main/docs/deep_probability_estimation.pdf)

## Pre-Trained Models and Code
Please visit [our GitHub page](https://github.com/jackzhu727/deep-probability-estimation/) for data, pre-trained models, code, and instructions on how to use the code. 
