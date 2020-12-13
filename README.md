# EEG-BCI
**Classification of EEG signals for brain-computer interface movements left/right or neither**

If you don't want to read a lot check '2stage_algoritham.ipynb' for the best classifier.

In this work EEG signals, corresponding to the thought of the directions left, right and neither were analyzed and classified. Due to the small number of examples and the large number of attributes, special attention was given to the selection of important channels and attributes.

Spectral characteristics of the signals were derived via the Welch method. The channels that record important information about the tasks were selected by ranking, followed by step-by-step selection. The k nearest neighbor algorithm was used to estimate the subsets of channels.

A recurrent neural network with a GRU cell was used to classify the raw signals. Its accuracy was comparable to a random forecast. The pre-processing of EEG signals and the number of examples are essential for their correct classification.

Spectral characteristics of the signals that are independent of the class according to the χ-square test and characteristics that correlate with each other were removed. The critical level of correlation for each of the attributes depends on the reliability with which the attribute is independent of the class. The dimensionality of the data was then reduced by the principal component method. Space reduction is necessary due to the large number of attributes that contribute noise and the large number of correlations in the data.

The processed signals were classified by different algorithms: classification tree, k nearest neighbors, supporting vector machines with RBF kernel and without kernel.
Three different support vector classification strategies were applied. These are the strategies "one vs all", "one vs one" and classification "movement / word", followed by classification "left / right" for examples of class movement.

The best accuracy was achieved by a system of two supporting vector machines with RBF kernel, where first the example is classified as "motion / word" and then as "left / right" if it belongs to the class "motion". The main advantage of this system is the smaller number of classifiers. Despite the small number of examples and the lack of ability for the user to adapt to the system, the achieved accuracy is much greater than a random forecast.

**Data**

row signals from:
BCI Competition III
Data set V ‹mental imagery, multi-class›
