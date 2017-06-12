# iOS-Digit-Recognizer

iOS digit handwriting classifier. Leverages Apple CoreML and Vision frameworks introduced at WWDC17. Using the MNIST dataset, a Convolutional Neural Network was built and trained in Keras. Apple's CoreMLTools was used to convert to a CoreML Model to interface with the frameworks.

```mnist.py``` contains the Keras model, including loading the dataset and training. ```coreMLConverter.py``` loads Keras model and converts into a .mlmodel file.

<img src="Screenshot.png" height="700" />
