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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, recall_score, confusion_matrix, multilabel_confusion_matrix


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



    @staticmethod
    def select_features(df, n_features_to_select):
        # Split data into training and testing sets
        X = df.drop(columns=['Attack Type'])
        y = df['Attack Type']

        # Filter method: SelectKBest using chi-squared test
        k = 30
        selector = SelectKBest(score_func=chi2, k=k)
        X_filtered = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

        # Wrapper method: Recursive Feature Elimination using Random Forest Classifier
        estimator = RandomForestClassifier(n_estimators=10, random_state=0)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        X_wrapped = rfe.fit_transform(X_filtered, y)
        selected_features = selected_features[rfe.get_support()]

        # Create a new dataframe with selected features and target variable
        X_selected = pd.DataFrame(X_wrapped, columns=selected_features)
        y_selected = y.copy()
        selected_data = pd.concat([X_selected, y_selected], axis=1)
        newdf = selected_data.copy()

        return newdf


    @staticmethod
    def plot_feature_importance(df, target_col, fig_name):
        
        # del df["Unnamed: 0"]
        # encode non-numeric features using LabelEncoder
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])


        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # split into features and target

        y = y.astype(int)

        model = RandomForestClassifier()
        model.fit(X, y)

        importance = model.feature_importances_
        feature_names = X.columns

        plt.figure(figsize=(6,8))
        plt.barh(feature_names, importance, color='black')
        plt.xlabel('Feature Importance', fontsize=16)
        plt.ylabel('Feature Name', fontsize=16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 12)
        plt.savefig(fig_name)
        plt.show()








    @staticmethod
    def window_data(data, features, targets, window_size):
        # Create empty lists to hold the input and target data
        input_data = []
        target_data = []
        indices = np.array([window_size*j for j in range(len(data)//window_size)])
        # Iterate over the dataframe to create the input and target data
        for i in indices: #range((len(data) - window_size)//window_size):
            input_data.append(data[features].values[i: i + window_size])
            target_data.append(data[targets].values[i: i + window_size])

        # Convert the input and target data to numpy arrays
        input_data = np.array(input_data)
        target_data = np.array(target_data)
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, input_data, target_data

    @staticmethod
    def plot_tf_model(model,  name):
        
        return plot_model(model,name)

    @staticmethod
    def save_model(model, name):
    
        return model.save(name)

    @staticmethod
    def load_model(name: str):
        
        model = tf.keras.models.load_model( name)
        return model


    # Define a function to plot the learning curve
    @staticmethod    
    def plot_learning_curve(history):
        plt.plot(history.history['loss'], 'k-')
        plt.plot(history.history['val_loss'], 'k-.')
        #plt.title('Model loss')
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    # Define a function to plot predicted vs true values
    @staticmethod
    def plot_predicted_vs_true(predicted, true):
        plt.plot(predicted)
        plt.plot(true)
        plt.title('Predicted vs True Values')
        plt.ylabel('Value')
        plt.xlabel('Timestep')
        plt.legend(['Predicted', 'True'], loc='upper right')
        plt.show()

    # Define a function to plot a scatter plot of predicted vs true values
    @staticmethod
    def scatter_plot_predicted_vs_true(predicted, true):
        plt.scatter(predicted, true)
        plt.title('Predicted vs True Values')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    # Define a function to plot a bar chart of the evaluation metrics
    @staticmethod
    def plot_metrics(y_test, y_pred):
        # Calculate the evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create a bar chart of the metrics
        met = {'MAE': mae, 'MSE': mse, 'R2': r2} #'MAPE': mape
        plt.bar(range(len(met)), list(met.values()), align='center', color="black")
        plt.xticks(range(len(met)), list(met.keys()))
        plt.xlabel('Metric', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        #plt.title('Evaluation Met')
        plt.show()


    @staticmethod
    def plot_interp(y_true, y_pred, order=2):
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the true and predicted values with markers
        ax.scatter(range(len(y_true)), y_true, label='True')
        ax.scatter(range(len(y_pred)), y_pred, label='Predicted')
        
        # Create interpolation functions for the true and predicted values
        f_true = interp1d(range(len(y_true)), y_true, kind='cubic', bounds_error=False)
        f_pred = interp1d(range(len(y_pred)), y_pred, kind='cubic', bounds_error=False)
        
        # Plot the interpolated lines for the true and predicted values
        x_interp = np.linspace(0, len(y_true)-1, num=1000)
        ax.plot(x_interp, f_true(x_interp))
        ax.plot(x_interp, f_pred(x_interp))
        
        # Add labels and legend
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Show the plot
        plt.show()

    @staticmethod
    def evaluate_model_performance(ids_model, X_train, X_test, y_train, y_test, epochs):
        
        # build the model
        ids_model.build()
        _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.25)
        # # define callbacks
        # early_stop = EarlyStopping(monitor='val_loss', patience=5)
        # tensorboard = TensorBoard(log_dir=path+'/logs', histogram_freq=0, write_graph=True, write_images=True)
        # model_checkpoint = ModelCheckpoint(filepath=path+'/checkpoint/best_model.h5', save_best_only=True, save_weights_only=False)

        # train the model
        starttrain = time.time()
        historyIds = ids_model.train(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[])
        training_time = time.time() - starttrain
        # Make predictions using the model
        starttest = time.time()
        y_pred = ids_model.predict(X_test)
        testing_time = time.time() - starttest

        yt = 9*y_test.reshape(-1,1)[:,0] 
        yp = 9*y_pred.reshape(-1,1)[:,0]
        yt = yt.round()
        yp = yp.round()
        yt = yt.astype(int)
        yp = yp.astype(int)
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp)
        r2 = r2_score(yt, yp)

        accuracy = accuracy_score(yt, yp)
        conf_matrix = multilabel_confusion_matrix(yt, yp)
        tp = conf_matrix[:,0,0]
        fp = conf_matrix[:,0,1]
        fn = conf_matrix[:,1,0]
        tn = conf_matrix[:,1,1]
        # #metrics['Sensitivity'] = tp/(tp+fn) if fn != 0 else 1.0
        # #metrics['Specificity'] = tn/(tn+fp) if fn != 0 else 1.0
        sensitivity = tp.mean()/(tp.mean()+fn.mean()) if fn.mean() != 0 else 1.0
        specificity = tn.mean()/(tn.mean()+fp.mean()) if fp.mean() != 0 else 1.0

        f1 = f1_score(yt, yp, average="macro")
        recall = recall_score(yt, yp, average="macro")

        #============================================================
        startval = time.time()
        y_predV = ids_model.predict(X_val)
        validation_time = time.time() - startval

        ytV = 9*y_val.reshape(-1,1)[:,0] 
        ypV = 9*y_predV.reshape(-1,1)[:,0]
        ytV = ytV.round()
        ypV = ypV.round()
        ytV = ytV.astype(int)
        ypV = ypV.astype(int)
        maeV = mean_absolute_error(ytV, ypV)
        mseV = mean_squared_error(ytV, ypV)
        mapeV = mean_absolute_percentage_error(ytV, ypV)
        r2V = r2_score(ytV, ypV)

        accuracyV = accuracy_score(ytV, ypV)
        conf_matrixV = multilabel_confusion_matrix(ytV, ypV)
        tpV = conf_matrix[:,0,0]
        fpV = conf_matrix[:,0,1]
        fnV = conf_matrix[:,1,0]
        tnV = conf_matrix[:,1,1]
        # #metrics['Sensitivity'] = tp/(tp+fn) if fn != 0 else 1.0
        # #metrics['Specificity'] = tn/(tn+fp) if fn != 0 else 1.0
        sensitivityV = tpV.mean()/(tpV.mean()+fnV.mean()) if fnV.mean() != 0 else 1.0
        specificityV = tnV.mean()/(tnV.mean()+fpV.mean()) if fpV.mean() != 0 else 1.0

        f1V = f1_score(ytV, ypV, average="macro")
        recallV = recall_score(ytV, ypV, average="macro")
        #============================================================
        metrics = {
        "MAE": [mae,maeV],
        "MSE": [mse,mseV],
        "R2": [r2,r2V],
        "Accuracy score": [accuracy, accuracyV],
        "Sensitivity": [sensitivity, sensitivityV],
        "Specificity": [specificity, specificityV],
        "F1": [f1, f1V],
        "Recall": [recall, recallV],
        "Training time": [training_time, training_time],
        "Testing time": [testing_time, validation_time],
        "TP": [tp.mean(),tpV.mean()],
        "FP": [fp.mean(),fpV.mean()],
        "TN": [tn.mean(),fpV.mean()],
        "FN": [fn.mean(),fnV.mean()],
        "cm": [conf_matrix, conf_matrixV]
        }
        # Create a bar chart of the metrics
        # Create a bar chart of the metrics
        plot_learning_curve(historyIds)
        #plot_interp(yt, yp)
        plot_metrics(yt, yp)
        return historyIds, metrics, [conf_matrix, conf_matrixV], yt, yp

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)