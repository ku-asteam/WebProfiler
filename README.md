# WebProfiler
Navigation Prediction Model in WebProfiler

## Introduction

Navigation Prediction Model in WebProfiler is an implementation of *a*STEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT; <https://asteam.korea.ac.kr>) 's Web navigation prediction model using recurrent neural network (RNN), long short-term memory (LSTM), and gated recurrent unit (GRU) based on TensorFlow. The function of this software is to predict the application likely to be navigated next using deep learning techniques based on the collected user interaction data of navigation and click events.

## Requirements and Dependencies

* *TensorFlow*: The versions above `1.14.0` are recommended
* *User Interaction Data* after preprocessed and transformed into the form suitable for training prediction models
* Other Python packages should be installed and imported properly

## Instructions

* Build Instructions of `ccp-agent`
  1. Open a terminal
  2. Install the Go tools: Please refer <https://golang.org/doc/install>.
  3. Following 'How to Write Go Code' <https://golang.org/doc/code.html>, organize your Go workspace
  4. Go to `src` directory of your Go workspace: `cd $GOPATH/src`
  5. Make directories `github.com/mit-nms`: `mkdir -p github.com/mit-nms`
  6. Go to `mit-nms` directory: `cd github.com/mit-nms`
  7. Clone `ccp-agent`'s Git repository: `git clone https://github.com/mit-nms/ccp`
  8. Go to `ccp-agent`'s Git repository: `cd ccp`
  9. Install required dependencies of `ccp-agent`: `go get ./...`
  10. Build `ccp-agent`: `make`

* Build Instructions of `ccp-chromium`
  1. Open a terminal and go to an appropriate directory (We assume the directory is user's home directory `~`.)
  2. Clone this Git repository: `git clone https://github.com/ku-asteam/ccp-chromium.git`
  3. Go to the local Git repository: `cd ccp-chromium`
  4. Build the code with `make`: `make all`

* How to Run (In Case of TCP Reno Congestion Algorithm)
  1. `$GOPATH/src/github.com/mit-nms/ccp/ccpl --datapath=udp --congAlg=reno`
  2. `~/ccp-chromium/testccp`
