Deep Attention Model
====================

The Deep Attention Model is a package that provides an implementation of a deep learning model with attention mechanism for classification tasks. It utilizes LSTM and multi-head attention layers to capture temporal dependencies and highlight relevant features in the input data.

Installation
------------

To install the Deep Attention Model package, you can use pip:

bash

```bash
pip install deep_attention_model
```

Usage
-----

### Initializing the Deep Attention Model

First, you need to initialize the Deep Attention Model by providing a path to the working directory where data, images, and models will be stored during calculations. This step allows the package to import all required dependencies and create the necessary folders.

python

```python
import deep_attention_model as da

# Specify the path to the working directory
mypath = '/path/to/working/directory'

# Initialize the Deep Attention Model
da.DAModel.initialize(path=mypath)
```

### Data Preprocessing and Splitting

The Deep Attention Model provides a convenient method for data preprocessing and splitting into train and test sets. The `get_samples` method handles missing data, duplicates, infinities, and also numerizes categorical features.

python

```python
import pandas as pd

# Load the data into a Pandas DataFrame
data = pd.read_csv("data.csv")

# Perform data preprocessing and split into train and test sets
X_train, X_test, y_train, y_test = da.DAModel.get_samples(dataframe=data, target='label', test_size=0.2, samples=n, stratify=True)
```

### Building and Training the Model

After preprocessing the data, you can build and train the Deep Attention Model using the `DAModel` class. Specify the model architecture and compile it with the desired optimizer, loss function, and metrics. Then, train the model on the training data.

python

```python
# Define the model architecture
model = da.DAModel(look_back=seq_length, n_features=X_train.shape[1], n_outputs=num_classes, seq_length=seq_length, num_heads=8)

# Build the model
model.build()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

### Evaluating Model Performance

To evaluate the performance of the trained model on the test data, you can use the `evaluate_model_performance` method in the `DAprocessor` class. It calculates metrics such as accuracy, precision, recall, and F1-score, and also generates a confusion matrix.

python

```python
from deep_attention_model import DAprocessor

# Evaluate model performance
metrics, confusion_matrix, y_true, y_pred = DAprocessor.evaluate_model_performance(model, X_test, y_test)

# Print the metrics
print(metrics)

# Plot the confusion matrix
DAprocessor.plot_confusion_matrix(confusion_matrix, file_name='confusion_matrix.png')
```

### Saving and Loading the Model

You can save the trained model to a file and load it later for inference or further training using the `save_model` and `load_model` methods.

python

```python
# Save the model
model.save_model(filepath='model.h5')

# Load a pre-trained model
loaded_model = da.DAModel()
loaded_model.load_model(filepath='model.h5')

# Make predictions using the loaded model
predictions = loaded_model.predict(X_test)
```

Examples
--------

You can find example scripts demonstrating the usage of the Deep Attention Model in the [examples](examples/) directory.

Contributing
------------

Contributions to the Deep Attention Model project are welcome! If you have any ideas, bug reports, or feature requests, please feel free to open an issue on the project's GitHub repository. If you'd like to contribute code, you can fork the repository, make your changes, and submit a pull request. Please ensure that your code follows the project's coding style and includes appropriate tests.

License
-------

The Deep Attention Model package is released under the [MIT License](LICENSE).

Credits
-------

The Deep Attention Model package is developed and maintained by the Deep Learning team at the Research Centre for AI and Robotics, Near East University. We would like to acknowledge the contributions of the open-source community and the libraries that we have used to build this package.

*   TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
*   Keras: [https://keras.io/](https://keras.io/)
*   NumPy: [https://numpy.org/](https://numpy.org/)
*   Pandas: [https://pandas.pydata.org/](https://pandas.pydata.org/)
*   Matplotlib: [https://matplotlib.org/](https://matplotlib.org/)
*   Seaborn: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

Contact
-------

If you have any questions or inquiries regarding the Deep Attention Model package, please contact our team at [mercel.vubangsi@aiiot.website](mailto:mercel.vubangsi@aiiot.website).

---

Thank you for choosing the Deep Attention Model package! We hope it helps you in your classification tasks. If you have any feedback or suggestions for improvement, we would love to hear from you. Happy coding!