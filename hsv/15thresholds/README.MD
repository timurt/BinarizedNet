Cifar10 to HSV colorspace, 9 thresholds
===================

- binaryInputsWeightsThreshold.py - trains and stores 9 models 
using binarized VGG11 model
- binaryInputsWeightsThresholdTest.py - Helper class, displays
accuracies of each models.
- binaryInputsWeightsLinear.py - trains common classifier for 
9 pretrained models
- binaryInputsWeightsVoting.py - voting algorithm
- binaryNet.py - Helper class

0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360

Model accuracies
-------------
| Model No | Accuracy rate | Threshold range |
| ------------- |:-------------:|:-------------:|
| 1| 16.36 | 0 - 24|
| 2| 19.23 | 25 - 48|
| 3| 15.97 | 49 - 72|
| 4| 13.92 | 73 - 96|
| 5| 12.18 | 97 - 120|
| 6| 13.50 | 121 - 144|
| 7| 14.74 | 145 - 168|
| 8| 16.91 | 169 - 192|
| 9| 19.96 | 193 - 216|
|10| 18.07 | 217 - 240|
|11| 16.69 | 241 - 264|
|12| 14.4 | 265 - 288|
|13| 13.89 | 289 - 312|
|14| 14.41 | 313 - 336|
|15| 14.49 | 337 - 360|



Final accuracies
-------------
| Strategy | Accuracy rate |
| ------------- |:-------------:|
| Removed last classifier from each model, big common classifier|    27.57  |