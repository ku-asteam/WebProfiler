# WebProfiler
Navigation Prediction Model for General Web Applications

## Introduction

Navigation Prediction Model in WebProfiler is an implementation of *a*STEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT; <https://asteam.korea.ac.kr>) 's Web navigation prediction model using recurrent neural network (RNN), long short-term memory (LSTM), and gated recurrent unit (GRU) based on TensorFlow. The function of this software is to predict the application likely to be navigated next using deep learning techniques based on the collected user interaction data of navigation and click events.

## Requirements and Dependencies

* *TensorFlow*: A version above `1.14.0` is recommended
* *User Interaction Data* after preprocessed and transformed into the form suitable for training prediction models
* Other Python packages should be installed and imported properly

## Instructions

* Prepare the preprocessed user interaction data in the `webprofiler` directory
* Set the parameters for training, including deep learning model, range of random seeds, number of units/layers, learning rate, number of epochs, batch size, activation function, optimizer
* Execute `run.py` to initiate iterative training and check the results for test set
* See the example of executing WebProfiler:
```
==================== 20191111_132326: deep-learning ====================
 ***** RANDOM_SEED 42 *****
Read metadata and dictionary files... done!
Read input and output files... done!

* Variables for training gru
 - len(X_train)|len(X_valid)|len(X_test): 16839|3608|3609
 - n_steps (# of time steps): 10
 - n_inputs (dimension of inputs): 64
 - n_neurons (# of neurons at each cell): 150
 - n_layers (# of cells): 2
 - n_outputs (dimension of outputs): 200
 - learning_rate (learning rate for optimizer): 0.010000
 - n_epochs (# of epochs to iterate): 101
 - batch_size (# of samples in each batch): 100
 - activation: relu
 - initializer: He
 - optimizer: Momentum
 - TOP_K_THRESH (top K threshold for accuracy): 3

* Training
Epoch 0: Batch Precision/Recall/F-measure: 0.420000/0.684783/0.520661, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589
Epoch 1: Batch Precision/Recall/F-measure: 0.390000/0.657303/0.489540, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589
Epoch 2: Batch Precision/Recall/F-measure: 0.350000/0.617647/0.446809, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589
Epoch 3: Batch Precision/Recall/F-measure: 0.320000/0.585366/0.413793, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589
Epoch 4: Batch Precision/Recall/F-measure: 0.330000/0.596386/0.424893, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589
Epoch 5: Batch Precision/Recall/F-measure: 0.400000/0.666667/0.500000, Validation Precision/Recall/F-measure: 0.368348/0.636291/0.466589

...

Epoch 100: Batch Precision/Recall/F-measure: 0.960000/0.986301/0.972973, Validation Precision/Recall/F-measure: 0.616685/0.828369/0.707023
   >>> The training and validation set precision/recall/f-measure values as epochs are stored in .\webprofiler\result\20191111_132326\42_gru_train-valid-accuracy.txt
   >>> The checkpoint for the trained model is stored in .\webprofiler\checkpoint\20191111_132326\42_my_gru_model_final.ckpt

* Testing
[gru] Test Precision/Recall/F-measure: 0.626489/0.834215/0.715582
    >>> The testing set precision/recall/f-measure values is stored in .\webprofiler\result\20191111_132326\42_test-accuracy.txt

 ***** RANDOM_SEED 42 *****
==================== 20191111_132326: deep-learning ====================
```
