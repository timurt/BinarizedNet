Cifar10 to Grayscale, 9 thresholds
===================

https://colab.research.google.com/drive/1Zn57m4WA1p8yJXIcN50NUHSvhGAmHXNg#scrollTo=Fm2U119sn5Mb

- main.py - Camila's implementation with grayscaled Cifar10
- mainCifar.py - Camila's model without thresholding
- mainCifarVGG - binarized VGG11 model with grayscaled Cifar10
- binaryInputsWeightsThreshold.py - trains and stores 9 models 
using binarized VGG11 model
- binaryInputsWeightsThresholdTest.py - Helper class, displays
accuracies of each models.
- binaryInputsWeightsLinear.py - trains common classifier for 
9 pretrained models
- binaryInputsWeightsVoting.py - voting algorithm
- binaryNet.py - Helper class
- mainResNet.py - ResNet50 without modifications with 2 thresholds and 
grayscaled cifar10 data
- mainMultipeModels.py - Multiple separate models using Camila's implementation with 
grayscaled Cifar10
- mainMultipleModelsLinear.py - Common classifier for *mainMultipeModels.py*

main.py
-------------
| Number of thresholds | Accuracy rate |
| ------------- |:-------------:|
| 3     | 37.49 |
| 9     | 36.75      |
| 15 | 36.23 |
| 50 | 34.05 |

mainCifarVGG.py
--------------
| Number of thresholds | Accuracy rate |
| ------------- |:-------------:|
|   3  | 18.82  |

mainResNet.py
-------------
| Number of thresholds | Accuracy rate |
| ------------- |:-------------:|
| 2     | 62.84 |

mainMultipeModels.py
--------
| Number of thresholds | Accuracy rate |
| ------------- |:-------------:|
| 3     | 40.45 |
| 9     | 44.15 |
| 15     | 44.19 |
| 30     |  |
| 50     |  |
