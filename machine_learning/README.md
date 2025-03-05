# Machine Learning Transfer Study 

Conditions 

Video
- Original Video (Openface)
- Metahuman Video (Openface)
- Furhat Video (Openface)

Audio 
- Audio (Opensmile)

Audio + Video
- Audio + Original Video (Openface + Opensmile)
- Audio + Metahuman Video (Openface + Opensmile)
- Audio + Furhat Video (Openface + Opensmile)

## Supervised Learning

Suggested models: 
- SVM
- Logistic Regression

For all models I will apply an exhaustive grid search to find the best hyperparameters.

I will begin by training and evaluating the models separately for each condition. It could potentially be interesting 
to train a model on all the different video conditions and evaluate it together.

It could also be interesting to train a model on more data in the original condition and then try to run a prediction 
on the different conditions. We will most likely get much better results on the original condition, but we can argue that
humans have similar "training".

### Objectives 

- Compare the performance of the models on the different conditions.
- Use permutation importance to identify the most important features.
- Use Coefficients to identify how features contributes to the predictions. 

## Unsupervised Learning

Suggested models:

- PCA
- TSNE

Use tensorboard to visualize the results.

### Objectives

Explore high level differences between the different conditions for the video modality. 




