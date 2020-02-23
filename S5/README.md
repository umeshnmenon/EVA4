# This contains the solutions of Assignment 5 of EVA4. This assignment is about the step by step improvement in a DNN architecture to achive:
1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters

## Notebook 1
### Target:
   1. Get the basic structure right - All cells must run corretly for a DNN task.
   2. Add ReLU and BatchNorm in every convolution layer except at last before output layer
   3. Make the model lighter - With less than 10k parameter
   4. Add maxpooling at RF = 5 (by looking at the image)
   5. Add Gobal average pooling

### Result:
   1. Parameters: 7,600
   2. Best train accuracy: 99.13
   3. Best test accuracy: 99.07(13th epoch)

### Analysis:
    1. The model is good and light
    2. At 19th epoch train accuracy: 99.33 and test accuracy: 99.04 (we see overfitting)
    3. Need to use Regularization

## Notebook 2
### Target:
  1.Add Regularization - Dropout
  2.Add Dropout to each layer

### Result:
   1.Parameter:7,600
   2.Best train accuracy:98.34
   3.best test accuracy:99.02(14th epoch)

### Analysis:
   1. The model is not over-fitting
   2. The model is under fitting because we using droupout to every layer and making model to train hard
   3. Add a layer after Gobal average pooling.Adding dense layer after GAP

## Notebook 3
### Target:
    1. Increase capacity of model.
    2. Add a layer after Gobal average pooling.Adding dense layer after GAP

### Result:
    1. Parameters:9,660
    2. Best train accuracy:98.84
    3. Best test accuracy:99.39(13th epoch)

### Analysis:
   1: The model is not over fitting.
   2. At (18th epoch) train accuracy:98.92 and test accuracy:99.46
   3. But we are not getting 99.40+ below 15th epoch
   4. Increase the capacity of model
## Notebook 4
### Target:
    1. Increase the capacity of model.

### Result:
   1. Parameters:9,980
   2. Best train accuracy:98.89
   3. Best test accuracy:99.37(15th epoch)

### Analysis:
   1. We got test accuracy: 99.47 and 99.45 (17th and 19th epoch).
   2. The model is not over fitting .
   3. By seeing images all numbers are not in same shape.so,Add rotation
## Notebook 5
### Target:
   1. Add rotation 

### Result:
  1. Parameters:9,980
  2. Best train accuracy:98.67
  3. Best test accuracy:99.47(14th epoch)

### Analysis:
   1. The model is works.
   2. The model is underfitting.
   3. We can see 99.4+ repeated more than twice.
   4. The test accuracy increased because of RandomRotation 
### Team
- Amit Jaiswal
- Bharath Kumar Bolla
- Raman Shaw
- Umesh Menon
