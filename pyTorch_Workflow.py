import torch
from torch import nn #nn contains all of pyTorch building blocks of neural networks.
import matplotlib.pyplot as plt

what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together"}

# print(torch.__version__)

# 1. Data - preparing and loading

# Data can be almost anything in machine learning e.g. Excel, text, image, video, audio and DNA
#ML is two parts:
# 1. get data into a numerical representation
# 2. Build a model to learn patterns in the numerical representation

#Create known parameters
weight = 0.7 #Weight controls how much influence one input has on the output.
bias = 0.3 #Bias is like a baseline adjustment - it shifts the output up or down regardless of the input.

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) #Add dimension
y = weight * X + bias #Linear regression.

#print(X[:10], y[:10], len(X), len(y))

#Splitting data into training and test sets.

#Create a train/test split

train_split = int(0.8 * len(X))
print(train_split)

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

#Visualizing the data

def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    prediction=None):
    print("Plot prediction called.")
    plt.figure(figsize=(10, 7))

    #plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    #plt.show()

    #Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    #plt.show()

    #Are there predictions?

    if prediction is not None:
        #plot the predictions if they exist
        plt.scatter(test_data, prediction, c="r", s=4, label="prediction")
        #plt.show()

    plt.legend(prop={"size": 14})
    plt.show()

plot_prediction()


# 2. Build model

#   1. Start with random values (weight & bias)
#   2. Look at training data and adjust the random values to better represent (or get closer to) the ideal values (the weight & bias values we used to create the data)

#    how does it do so?
#    1. gradient descent
#    2. Back propagation

#Create linear regression model class
class LinearRegressionModel(nn.modules): #<- almost everything in PyTorch inherit from nn.module Base class for all neural network module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # X is the input data
        return self.weights * x + self.bias #This is linear regression.











