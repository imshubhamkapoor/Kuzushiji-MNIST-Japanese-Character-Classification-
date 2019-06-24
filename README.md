# Kuzushiji_MNIST_Japanese_Character_Classification

## About the problem
Recorded historical documents give us a peek into the past. We are able to glimpse the world before our time; and see its culture, norms, and values to reflect on our own. Japan has very unique historical pathway. Historically, Japan and its culture was relatively isolated from the West, until the Meiji restoration in 1868 where Japanese leaders reformed its education system to modernize its culture. This caused drastic changes in the Japanese language, writing and printing systems. Due to the modernization of Japanese language in this era, cursive Kuzushiji (traditional) (くずし字) script is no longer taught in the official school curriculum. Even though Kuzushiji had been used for over 1000 years, most Japanese natives today cannot read books written or published over 150 years ago.

The result is that there are hundreds of thousands of Kuzushiji texts that have been digitised but have never been transcribed, and can only currently be read by a few experts. Here is given dataset of Kuzushiji-MNIST made by taking handwritten characters from these texts and preprocessing them in a format similar to the MNIST dataset, to create easy to use benchmark datasets that are more modern and difficult to classify than the original MNIST dataset.

## Introduction to Project

In this project, you'll train a convolutional neural network to classify and recognize different types of Kuzushiji characters. We'll be using [this dataset](https://www.kaggle.com/anokas/kuzushiji) of 10 categories of Kuzushiji characters to train our model.
The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Visualization of samples from the dataset
- Train the Convolutional Neural Network on your dataset
- Use the trained model to predict new characters

Each part will be implemented in Jupyter Notebook.

## Dataset Description

Kuzushiji-MNIST is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in a NumPy format. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 classes of Hiragana (total 49 classes of Japanese syllable system) when creating Kuzushiji-MNIST.

Some examples of Kuzushiji-MNIST with the first column being the modern Hiragana counterpart: Kuzushiji-MNIST examples

![](https://raw.githubusercontent.com/rois-codh/kmnist/master/images/kmnist_examples.png)

## Files Description

- **main_file.ipynb** It contains the full code and is used to build the model using the jupyter notebook. It can be used independently to see how the model works.
- **kmnist_classmap.csv** It is used in ipynb file to map from class IDs to unicode characters for Kuzushiji-MNIST.
- **kmnist-train-images.npz** It contains the training dataset of 60,000 images (28x28 grayscale) provided in a NumPy format.
- **kmnist-test-images.npz** It contains the test dataset of 10,000 images (28x28 grayscale) provided in a NumPy format.
- **kmnist-train-labels.npz** It contains the training labels of 60,000 images provided in a NumPy format for training dataset.
- **kmnist-test-labels.npz** It contains the test labels of 10,000 images provided in a NumPy format for test dataset.

**NOTE:** kmnist-[train/test]-[images/labels].npz: These files contain the Kuzushiji-MNIST as compressed numpy arrays, and can be read with: arr = np.load(filename)['arr_0']. We recommend using these files to load the dataset.

## Installation
The Code is written in Jupyter Notebook.

Additional Packages that are required are: Numpy, Pandas, MatplotLib, Pytorch, and PIL. You can donwload them using pip

`pip install numpy pandas matplotlib pil`

In order to intall Pytorch head over to the [Pytorch](https://pytorch.org/get-started/locally/) website and follow the instructions given.

## GPU/CPU

As this project uses deep CNNs, for training of network you need to use a GPU. However after training you can always use normal CPU for the prediction phase.

## License
[MIT License](https://github.com/imshubhamkapoor/Kuzushiji_MNIST_Japanese_Character_Classification/blob/master/LICENSE)

## Author
Shubham Kapoor
