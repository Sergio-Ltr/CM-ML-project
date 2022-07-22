from logger import MLPLogger
from model import Model
from layers import FullyConnectedLayer
from network import Network
from training import Training
from utils import multiline_plot, log


class MLP(Model): 
    def __init__(self, name, architecture, hyperparameters = [], verbose=True, make_folder = True): 
        units = architecture.units
        activations = architecture.activations
        initializations = architecture.initializations

        self.structural_hps = {} 
        self.training_hps = {} 
        
        for hp in hyperparameters: 
            getattr(self, 'training_hps' if hp.training else 'structural_hps')[hp.key] = hp.value()
        
        self.network = Network(architecture.loss, **self.structural_hps )

        for i in range(len(units) - 1):
            self.network.add(FullyConnectedLayer(units[i], units[i+1], activations[i], initializations[i]))

        super().__init__(name + '_MLP', MLPLogger(name, architecture, hyperparameters, verbose), make_folder)

        self.training_algorithm = Training(self.network, self.training_hps, logger=self.logger)
        self.trained = False
    

    def predict(self, X):
        output = []

        for i in range(len(X)):
            output.append(self.network.forward_propagation(X[i], inference=True))

        return output


    def train(self, X_TR, Y_TR, X_VAL, Y_VAL, metric, verbose = True, plot_folder=None): 
        self.logger.summary()

        tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist = self.training_algorithm(
            X_TR, Y_TR, X_VAL, Y_VAL, metric=metric, verbose=verbose
        )

        history = [tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist]

        self.plot_training_curves(history[0:2], self.network.loss.name, plot_folder)
        self.plot_training_curves(history[2:4], metric.name, plot_folder)

        self.tr_loss = tr_loss_hist[-1]
        self.val_loss = val_loss_hist[-1]
        
        self.tr_metric = tr_metric_hist[-1]
        self.val_metric = val_metric_hist[-1]
        
        self.trained = True

        self.logger.training_results(self.tr_loss, self.tr_metric, metric)
        self.logger.validation_results(self.val_loss, self.val_metric, metric)


    # TODO move this to the training module.    
    def plot_training_curves(self, history, metric_name, folder ):
        legend_names = ["TR", "VL"] if len(history) == 2 else ["TR"]
            
        multiline_plot(
            title = f"{self.name}_{metric_name}",
            legend_names = legend_names,
            histories=list(history),
            ylabel=metric_name, xlabel="Epochs", 
            showlegend=True, showgrid=True, alternateDots=True,
            savefile=f"{metric_name}_TR", prefix = self.path if folder is None else folder 
        )


    def save(self, custom_path = None):
        """ Saves the neural network in a pickle """
        super().save('MLP', custom_path)
