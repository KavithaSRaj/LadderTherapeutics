# LadderTherapeutics
Coding challenge for interview - 
A Neural network classifier is implemented to classify Fashion MNIST dataset 

## **Pre-requisities:**
Please ensure below packages are installed before running this code\
\
-> Torch\
-> Torchvision\
-> Torchinfo\
-> Torcheck\
-> Tensorboard\
\
Version used: python = 3.9.7, torch = 1.11.0
Note: you can install these packages using "pip install <package_name>" at the terminal or using conda\

## **Configuration:**
All the hyperparameters and other variables for the model are stored in **config.yaml** file\
you can update the config.yaml file using any editor\

## **Code execution:**
Terminal: run below commands\
\
***python fashion_mnist_model_kavitha.py --config config.yaml***\
\
*This is the main file which loads the data, trains the model and tests the model\
*Note: Code for testing is in a model_eval() function which can be used separately for furture use \
*Plots and print statements are provided to see how the code is executing and results at each stage\
*Torcheck is used to do basic sanity checks

\
***tensorboard --logdir=runs --host=127.0.0.1 --port=6006*** \
\
*To view tensorboard in local host, excute this command at the terminal and open the link in http://127.0.0.1:6006 \
*Check Section "SCALARS" for - Training / testing - loss and accruracy\
*Check Section "GRAPHS" for - Classifier model graph which shows each layer of the neural network wit its inputs and outputs\
*Check Section "IMAGES" for -samaple images from the dataset\
*Check Section "PR CURVES" for - Precision Recall plots for each class\
