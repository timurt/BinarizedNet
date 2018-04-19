Cifar10 to HSV colorspace, 9 thresholds
===================

https://colab.research.google.com/drive/12J72fVUyOf6zuCbotsPIhx0EZeO84med

- binaryInputsWeightsThreshold.py - trains and stores 9 models 
using binarized VGG11 model
- binaryInputsWeightsThresholdTest.py - Helper class, displays
accuracies of each models.
- binaryInputsWeightsLinear.py - trains common classifier for 
9 pretrained models
- binaryInputsWeightsVoting.py - voting algorithm
- binaryNet.py - Helper class

Model accuracies
-------------
| Model No | Accuracy rate | Threshold range |
| ------------- |:-------------:|:-------------:|
| 1| 18.47 | 0 - 40|
| 2| 17.38 | 41 - 80|
| 3| 13.38 | 81 - 120|
| 4| 14.2  | 121 - 160|
| 5| 17.29 | 161 - 200|
| 6| 20.75 | 201 - 240|
| 7| 16.16 | 241 - 280|
| 8| 14.98 | 281 - 320|
| 9| 14.9  | 321 - 360|

Final accuracies
-------------
| Strategy | Accuracy rate |
| ------------- |:-------------:|
| Voting     | 24.95 |
| Fully-connected     | 26.29 |
| CNN     | 24.02      |
| Removed last classifier from each model, big common classifier|    27.62  |