# NeuralNetwork
Neural network for recognising handwritten numbers. Accuracy of 80% with 200 epoch.
Strongly inspired by Mr. Omar Aflak in "Neural Network from scratch in Python" article.
The Neural Network is built from scratch using numpy for numerical operations. 
A Neural Network for recognising handwritten numbers is made by layers, where the input layer gets the image of the number, and the output layer gives the final prediction.
Underneath the hood, in the "Hidden layers", the nerual network start by randomizing the prediction, and then by back-propagation it adjusts the parameters to make a more
precise prediction. The more times you iterate over the handwritten samples, the better it becomes.
The main file is "recognise.py"
