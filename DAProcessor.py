import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

