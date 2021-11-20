# FYS-STK4155-project2
This repository contains all code written for Project 2 in FYS-STK4155.

Ensure all packages in ´requirements.txt` are installed.

´notebooks/Reports´ contains notebooks specifically targeting each sub task of the project.
Refer to those for more details on the process.

The final written accompanying report can be found diretly in the repository as ´Project_2.pdf´.

Project structure
------------
 ```
FYS-STK4155-project2
│   README.md
│   requirements.txt                    <-- Package requirements/dependencies for this project
│
├───data                                <-- Data used for modelling
│   └───raw
│           SRTM_data_Norway_1.tif
│           SRTM_data_Norway_2.tif
│
├───figures                             <-- Figures and images produced
│   ├───classification
│   │   ├───logreg
│   │   └───mlp
│   └───regression
│       ├───mlp
│       └───sgdregressor
├───notebooks                           <-- Notebooks used in this project
│   ├───Reports                         <-- Notebooks specifically targeting a task
│   │       Part a.ipynb
│   │       Part_b_and_c.ipynb
│   │       Part_d.ipynb
│   │       Part_e.ipynb
│   │
│   └───Sketches                        <-- Sketches used during code development
├───src                                 <-- All source code for develop library
│   ├───data                            <-- Submodules for loading and creating data set
│   │       create_dataset.py
│   │
│   ├───modelling                       <--The core of this project, i.e the models
│   │       linreg.py                   <--Linear regression, with and without SGD
│   │       logreg.py                   <--Logistic regression
│   │       nn.py                       <--Neural Network
│   │       _functions.py               <--Selected functions used by above models
│   │       _sgdBase.py                 <--Base SGD which all SGD based models inherit from
│   │
│   ├───model_evaluation                <--Scripts for evaluating models
│   │       metrics.py                  <--Metrics like MSE, R2 and accuracy
│   │       param_analysis.py           <--Scripts for conducting parameter analysis,
│   │                                      like gridsearch
│   │       resampling.py               <--Resampling scripts like cv and bootstrap
│   │
│   ├───processing                      <--Scripts for preprocessing of data
│   │       data_preprocessing.py
│   │
│   ├───utils                           <--Utility/helper functions
│   │       utils.py
│   │
│   └───visualization                   <--Visualization scripts
│           visualize.py
│
└───tests                               <--Tests
│       Activation function tests.ipynb
│
└───Project_2.pdf                       <--Final report
```        
