from activationfunctions import Sigmoid
from losses import BinaryCrossentropy
from layers import FullyConnectedLayer
from neuralnetwork import Network
from metrics import Accuracy
from utils import plot_and_save
from dataset_loader import load_monk

monk = 3
print("\n\n****TESTING NETWORK ON MONK" + str(monk))

# Training

# training set loading + preprocessing
X_TR, Y_TR,input_size = load_monk(monk, use_one_hot=True)

# training
net = Network("MONK" + str(monk), BinaryCrossentropy())
net.add(FullyConnectedLayer(input_size, 4, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(4, 4, Sigmoid(), initialization_func="xavier"))
net.add(FullyConnectedLayer(4, 1, Sigmoid(), initialization_func="xavier"))
net.summary()
history = net.training_loop(X_TR, Y_TR, epochs=1000, learning_rate=0.01, verbose=True, early_stopping=50)

# Model evaluation

# test set loading
X_TS,Y_TS, input_size = load_monk(monk, use_one_hot=True, test=True)

# evaluating
accuracy = Accuracy().compute(net, X_TS, Y_TS)
print("Accuracy on the test set: {:.4f}%".format(accuracy))

# plotting data
plot_and_save(title="MONK3 model evaluation", history=history, ylabel="Loss", xlabel="Epochs", savefile="MONK3TEST")

# saving the net
net.savenet("models/MONK3TESTED_1L_10U_0.01LR_xavier.pkl")
