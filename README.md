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
* An example of executing WebProfiler:
```python
print("hello");
```
