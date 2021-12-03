## Deep Probability Estimation

This website contains results, code and pre-trained models from [Deep Probability Estimation](https://arxiv.org/abs/2111.10734) by Sheng Liu\*, Aakash Kaku\*, Weicheng Zhu\*, Matan Leibovich\*,  Sreyas Mohan\*, Boyang Yu, Laure Zanna, Narges Razavian, Carlos Fernandez-Granda [\* - Equal Contribution].

## What is probability estimation and why is it needed?
Reliable probability estimation is of crucial importance in many real-world applications where there is inherent uncertainty, such as weather forecasting, medical prognosis, or collision avoidance in autonomous vehicles. Probability-estimation models are trained on observed outcomes ( <img src="https://latex.codecogs.com/gif.latex?y_i" /> ) (e.g. whether it has rained or not, or whether a patient has died or not), because the ground-truth probabilities ( <img src="https://latex.codecogs.com/gif.latex?p_i" /> ) of the events of interest are typically unknown. The problem is therefore analogous to binary classification, with the important difference that the objective is to estimate probabilities ( <img src="https://latex.codecogs.com/gif.latex?\hat{p}" /> ) rather than predicting the specific outcome.

<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144637201-b9aed32f-f5e7-46f0-a4ef-0a9f2baa7a78.png" />
  <em> The probability-estimation problem. In probability estimation, we assume that each observed outcome <img src="https://latex.codecogs.com/gif.latex?y_i" /> (e.g. death or survival in cancer patients) in the training set is randomly generated from a latent unobserved probability <img src="https://latex.codecogs.com/gif.latex?p_i" /> associated to the corresponding data <img src="https://latex.codecogs.com/gif.latex?\vx_i" /> (e.g. histopathology images).Training (left): Only <img src="https://latex.codecogs.com/gif.latex?\vx_i" /> and <img src="https://latex.codecogs.com/gif.latex?\y_i" /> can be used for training, because <img src="https://latex.codecogs.com/gif.latex?\p_i" /> is not observed. Inference (right): Given new data <img src="https://latex.codecogs.com/gif.latex?\vx" />, the trained network <img src="https://latex.codecogs.com/gif.latex?f" /> produces a probability estimate <img src="https://latex.codecogs.com/gif.latex?\hat{p}\in [0,1]" />.</em>
</p>

## Seq2seq model predicting extremely fine-grained actions
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508990-195293f4-311b-469d-a2cd-92ff2841122e.png" />
</p>
Comparison of sequence-to-sequence (seq2seq) and segmentation models. The segmentation model outputs frame-wise action predictions, which can then be converted to a sequence
estimate by removing the duplicates. The seq2seq model produces a sequence estimate directly.

## Segmentation models cannot detect boundaries for extremely fine-grained actions
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508026-c03afa71-b454-484d-bddd-7f990372858e.png" />
</p>
Boundary accuracy achieved by the segmentation models vs duration of the actions for several datasets. Boundary-detection accuracy is inversely proportional to action duration.

## Performance Metric
In order to evaluate sequence predictions we use two metrics based on the Levenshtein distance: edit score (ES) and action error rate (AER) (inspired by the word-error rate metric used in speech recognition). The Levenshtein distance, L(G, P), is the minimum number of insertions, deletions, and substitutions required to convert a predicted sequence P to a ground-truth sequence G. For example, if G = [reach, idle, stabilize] and P = [reach, transport], then L(G, P) = 2 (transport is substituted for idle and stabilize is inserted). We have:
![image](https://user-images.githubusercontent.com/32464452/144508527-d6b8084a-0f45-46d4-aa0e-5e972ba18712.png)
where len(G) and len(P) are the lengths of the ground-truth and predicted sequence respectively. The edit score is more lenient when the estimated sequence is longer. In contrast, AER penalizes longer and shorter predictions equally. For example, if G = [reach, idle, stabilize], P1 = [reach,idle], and P2 = [reach, idle, stabilize, transport], then ES(G, P1) = 0.67 and ES(G, P2) = 0.75, but AER(G, P1) = AER(G, P2) = 0.33.

## Results
- **StrokeRehab dataset**
![image](https://user-images.githubusercontent.com/32464452/144508233-17f6920b-2c1a-44d0-a5ec-a1bfe1192bd2.png)
Results on StrokeRehab: Seq2seq outperforms segmentation-based approaches. We report mean (95% confidence interval) which is computed via bootstrapping.

- **Action-recognition benchmarks datasets**
![image](https://user-images.githubusercontent.com/32464452/144508275-282b8ede-9f09-4c8d-b72e-035984417f01.png)
Results on action-recognition benchmarks: Seg2seq, the seq2seq model which uses the output of a pretrained segmentation-based model, outperforms segmentation-based approaches.

- **Count of primitives for StrokeRehab dataset**
In stroke rehabilitation, action identification can be used for quantifying dose by counting functional primitives. The figure below shows that the raw2seq version of the seq2seq model produces accurate counts for all activities in the StrokeRehab dataset. Performance is particularly good for structured activities such as moving objects on/off a shelf, in comparison to less structured activities such as brushing, which tend to be more heterogeneous across patients
<p align="center">
  <img src="https://user-images.githubusercontent.com/32464452/144508718-6b122fe9-2fe8-4a47-9142-14733c6cd923.png" />
</p>
Comparison of ground-truth and predicted mean counts for the different activities in the StrokeRehab dataset. The relative error is very small for structured activities like moving objects on/off a shelf (Shelf), and larger for unstructured activities like brushing.


## Pre-Trained Models and Code
Please visit [our github page](https://github.com/aakashrkaku/seq2seq_hrar) for data, pre-trained models, code and instructions on how to use the code. 
