import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model


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


    @staticmethod
    def initialize(path):
        """ Initialize the DAModel
        :param path: Path to the working directory
        """
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_samples(dataframe, target, test_size, samples=None, stratify=None):
        """ Perform data preprocessing and split the data into train and test sets
        :param dataframe: Pandas DataFrame containing the data
        :param target: Name of the target variable column
        :param test_size: Fraction of the data to be used for testing
        :param samples: Number of samples to select from the data
        :param stratify: Array-like or None. If not None, data is split in a stratified fashion.
        :return: X_train, X_test, y_train, y_test
        """
        # Drop missing values and duplicates
        dataframe = dataframe.dropna().drop_duplicates()

        # Select the desired number of samples
        if samples is not None:
            dataframe = dataframe.sample(n=samples)

        # Extract features and target
        X = dataframe.drop(columns=[target]).values
        y = dataframe[target].values

        # Numerize categorical features if any
        label_encoders = {}
        for i in range(X.shape[1]):
            if isinstance(X[:, i][0], str):
                label_encoders[i] = LabelEncoder()
                X[:, i] = label_encoders[i].fit_transform(X[:, i])

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)

        return X_train, X_test, y_train, y_test