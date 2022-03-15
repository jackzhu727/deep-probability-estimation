## Deep Probability Estimation

This website contains results, code and pre-trained models from [Deep Probability Estimation](https://arxiv.org/abs/2111.10734) by Sheng Liu\*, Aakash Kaku\*, Weicheng Zhu\*, Matan Leibovich\*,  Sreyas Mohan\*, Boyang Yu, Laure Zanna, Narges Razavian, Carlos Fernandez-Granda [\* - Equal Contribution].

## What is probability estimation and why is it important?
Reliable probability estimation is of crucial importance in many real-world applications where there is inherent uncertainty, such as weather forecasting, medical prognosis, or collision avoidance in autonomous vehicles. 
<img align="left" width="512" height="512" src="https://user-images.githubusercontent.com/32464452/158394782-b9b7c4e4-544d-4fd0-82d9-9699e265b098.png">

Probability-estimation models are trained on observed outcomes ( <img src="https://latex.codecogs.com/gif.latex?y_i" /> ) (e.g. whether it has rained or not, or whether a patient has died or not), because the ground-truth probabilities ( <img src="https://latex.codecogs.com/gif.latex?p_i" /> ) of the events of interest are typically unknown. The problem is therefore analogous to binary classification, with the important difference that the objective is to estimate probabilities ( <img src="https://latex.codecogs.com/gif.latex?\hat{p}" /> ) rather than predicting the specific outcome.

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144637201-b9aed32f-f5e7-46f0-a4ef-0a9f2baa7a78.png" alt>
  <em> <br />The probability-estimation problem. In probability estimation, we assume that each observed outcome <img src="https://latex.codecogs.com/gif.latex?y_i" /> (e.g. death or survival in cancer patients) in the training set is randomly generated from a latent unobserved probability <img src="https://latex.codecogs.com/gif.latex?p_i" /> associated to the corresponding data <img src="https://latex.codecogs.com/gif.latex?x_i" /> (e.g. histopathology images).Training (left): Only <img src="https://latex.codecogs.com/gif.latex?x_i" /> and <img src="https://latex.codecogs.com/gif.latex?y_i" /> can be used for training, because <img src="https://latex.codecogs.com/gif.latex?p_i" /> is not observed. Inference (right): Given new data <img src="https://latex.codecogs.com/gif.latex?x" />, the trained network <img src="https://latex.codecogs.com/gif.latex?f" /> produces a probability estimate <img src="https://latex.codecogs.com/gif.latex?\hat{p}" /> in [0,1].</em>
</p>

## Evaluation metrics 
Probability estimation shares similar target labels and network outputs with binary classification. However, classification accuracy is __not__ an appropriate metric for evaluating probability-estimation models due to the inherent uncertainty of the outcomes. 

- **Metrics when ground truth probabilities are available**
 For synthetic datasets, we have access to the ground truth probability labels and can use them to evaluate performance. A reasonable metric is the mean squared error ( <img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" /> ) between the estimated probability and the ground truth probability.
- **Metrics when ground truth probabilities are not available**
  This is usually the case for most of the real-world datasets. There are several calibration metrics like ECE, MCE, KS-error or classification metrics like Brier score and AUC can be used to evaluate the performance of the model. But, from several metrics, which metric captures the true probability estimation performance? 

To answer this question, we use synthetic data to compare different metrics to the __gold-standard__  <img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" /> that uses ground-truth probabilities. Brier score is found to be highly correlated with <img src="https://latex.codecogs.com/gif.latex?\text{MSE}_p" />, in contrast to the classification metric AUC and the calibration metrics ECE, MCE and KS-Error.
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144640753-700c8858-09f1-4503-971f-aa73b0918c14.png" />
</p>

## Proposed Method: Calibrated Probability Estimation (CAPE)
We propose Calibrated Probability Estimation (CaPE), a novel technique which modifies the training process so that output probabilities are consistent with empirical probabilities computed from the data. CaPE outperforms existing approaches on most metrics on  synthetic and real-world data. The pseudo-code for our proposed approach can be seen below:
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144643132-a2557b03-43b2-48ad-949c-b42c2d5a0417.png" />
</p>

### Our proposed approach achieves two objectives:
- Avoids overfitting of the model.
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144643659-6537f6eb-ee52-46f5-ba0e-86e42dd90208.png" alt>
  <em> <br /> Comparison between the learning curves of cross-entropy (CE) minimization and the proposed calibrated probability estimation (CaPE), smoothed with a 5-epoch moving average. After an early-learning stage where both training and validation losses decrease, CE minimization overfits (first and second graph), with disastrous consequences in terms of probability estimation (third and fourth graph). In contrast, CaPE prevents overfitting, continuing to improve the model while maintaining calibration. </em>
</p>

- Improves calibration and discrimination performance of the model.
<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144642950-e477d168-793a-4d9e-818a-5e4c65b637c6.png" alt>
  <em> <br /> When trained on infinite data (i.e. resampling outcome labels at each epoch according to ground-truth probabilities), models minimizing cross-entropy are well calibrated (first column). The top row shows results for the synthetic Discrete scenario (top). The bottom row shows results for the Linear scenario (dashed line indicates perfect calibration). However, when trained on fixed observed outcomes, the model eventually overfits and the probabilities collapse to either 0 or 1 (second column). This is mitigated via early stopping (i.e. selecting the model based on validation cross-entropy loss), which yields relatively good calibration (third column). The proposed Calibration Probability Estimation (CaPE) method exploits this to further improve the model discrimination while ensuring that the output remains well calibrated.</em>
</p>


## Results
### Synthetic dataset - Face-Based Risk Prediction
  <p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144645100-8beb337d-3457-46c5-acd7-b8f88b849b1c.png" alt>
  <em> <br /> Our proposed approach outperforms existing approaches for different simulated scenarios.</em>
</p>

### Real-world datasets

- **Survival of Cancer Patients**: Based on the Hematoxylin and Eosin slides of non-small cell lung cancers from The Cancer Genome Atlas Program (TCGA), we estimate the the 5-year survival probability of cancer patients. 

- **Weather Forecasting**: We use the German Weather service dataset, which contains quality-controlled rainfall-depth composites from 17 operational Doppler radars. We use 30 minutes of precipitation data to predict if the mean precipitation over the area covered will increase or decrease one hour after the most recent measurement. Three precipitation maps from the past 30 minutes serve as an input.

- **Collision Prediction**: We use 0.3 seconds of real dashcam videos from __YouTubeCrash__ dataset as input, and predict the probability of a collision in the next 2 seconds.

On all the three real-world datasets, CaPE outperforms the existing calibration approaches (when compared on the Brier score which was found to capture the probability esitmation performace in the absence of the ground truth probabilities)

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144646458-3b68b90d-0cca-46b7-89ab-ba5dfea4584c.png" alt>
  <em> <br /> Our proposed approach outperforms existing approaches for different simulated scenarios.</em>
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/32464452/144646569-53fb0e4b-9a14-45e2-a6f7-d6a203dcd89a.png" alt>
  <em> <br /> Reliability diagrams for real-world data. Reliability diagrams computed on test data for cross-entropy minimization with early stopping, the proposed method (CaPE) and the best baseline for each dataset. Among all the methods, CaPE produces better calibrated outputs.</em>
</p>


## Pre-Trained Models and Code
Please visit [our GitHub page](https://github.com/jackzhu727/deep-probability-estimation/) for data, pre-trained models, code and instructions on how to use the code. 
