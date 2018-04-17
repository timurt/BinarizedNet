Cifar10 to Grayscale, 9 thresholds
===================

- main.py - Camila's implementation with grayscaled Cifar10
- mainCifar.py - Camila's model without thresholding
- mainCifarVGG - VGG11 model with grayscaled Cifar10
- binaryInputsWeightsThreshold.py - trains and stores 9 models 
using binarized VGG11 model
- binaryInputsWeightsThresholdTest.py - Helper class, displays
accuracies of each models.
- binaryInputsWeightsLinear.py - trains common classifier for 
9 pretrained models
- binaryInputsWeightsVoting.py - voting algorithm
- binaryNet.py - Helper class


main.py
-------------
| Number of thresholds | Accuracy rate |
| ------------- |:-------------:|
| 3     | 36.44 |
| 9     | 37.97      |
| 50 |       |