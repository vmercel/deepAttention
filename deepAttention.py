import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DAModel():
    """ Building the Bidirectional Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self, look_back, n_features, n_outputs, seq_length=24, lstm_units=128, bidirectional=True, mlp_units=[], mlp_dropout=0.2, dropout=0.2, num_heads=4):
        """ Initialization of the object
        """
        self.look_back = look_back
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lstm_units=lstm_units
        self.bidirectional=bidirectional
        self.mlp_units=mlp_units
        self.mlp_dropout=mlp_dropout
        self.dropout=dropout
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.best_model = None #self.best_model.compile(optimizer='adam', loss='mse')

    def build(self):
        """ Build the model architecture
        """
        inputs = keras.Input(shape=(self.look_back, self.n_features))
        
        # LSTM layer
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(self.dropout)(x)
        
        # Multilayer Attention layer
        for i in range(2):
            attention = layers.MultiHeadAttention(key_dim=self.n_features//4, num_heads=4)(x, x)
            x = layers.Dropout(self.dropout)(attention)
            x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention]))

        # LSTM layer
        x = layers.LSTM(128)(x)
        x = layers.Dropout(self.dropout)(x)
        
        # MLP layers
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        # output layer
        outputs = layers.Dense(self.n_outputs, activation="linear")(x)

        self.best_model = keras.Model(inputs, outputs)

    def restore(self, filepath):
        """ Restore a previously trained model
        """
        self.best_model = load_model(filepath)

    def train(self, X_train, y_train, epochs=200, batch_size=16, callbacks=[]):

        self.best_model.compile(optimizer='adam', loss='mse')
        history = self.best_model.fit(X_train, y_train, epochs=epochs, validation_split = 0.25, batch_size=batch_size, callbacks=callbacks)
        return history


    def predict(self, X_test):
        """ Predict the future
        :param X_test: feature vectors to predict [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        """
        return self.best_model.predict(X_test)         

    def plot_tf_model(self,  fname):
      path = "/content/drive/MyDrive/NEU/AIE/MERCEL THESIS/figs/"
      return plot_model(self.best_model,path + fname)


    def save_model(self, fname):
      path = "/content/drive/MyDrive/NEU/AIE/MERCEL THESIS/models/"
      return self.best_model.save(path + fname)  

    def print_model_summary(self):
        return self.best_model.summary()
    



class DAprocessor:
    @staticmethod
    def evaluate_model_performance(model, X_test, y_test):
        """ Evaluate the performance of the model on the test data
        :param model: The trained deep attention model
        :param X_test: Test data features
        :param y_test: Test data labels
        :return: metrics, confusion_matrix, y_true, y_predicted
        """
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }

        # Calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

        return metrics_dict, confusion_matrix, y_true, y_pred

    @staticmethod
    def plot_metric(data, metric_to_plot, file_name):
        """ Plot a bar plot of a specific metric
        :param data: Dictionary containing metrics and their values
        :param metric_to_plot: The metric to plot
        :param file_name: File name to save the plot
        """
        metric_value = data[metric_to_plot]

        plt.figure(figsize=(8, 6))
        plt.bar(metric_to_plot, metric_value)
        plt.title(metric_to_plot + ' Bar Plot')
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.savefig(file_name)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(data, file_name):
        """ Plot the confusion matrix as a heatmap
        :param data: Confusion matrix data
        :param file_name: File name to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(data, annot=True, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(file_name)
        plt.show()

    @staticmethod
    def plot_history(data, file_name):
        """ Plot training and validation loss
        :param data: Training history data
        :param file_name: File name to save the plot
        """
        loss = data.history['loss']
        val_loss = data.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(file_name)
        plt.show()

    @staticmethod
    def plot_metrics(data, file_name):
        """ Display metrics as a table and save it to a CSV file
        :param data: Metrics data
        :param file_name: File name to save the CSV file
        """
        metrics_df = pd.DataFrame(data)
        print(metrics_df)
        metrics_df.to_csv(file_name, index=False)

