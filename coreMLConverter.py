# Akhil Waghmare
# akhil@awlabs.technology

# import saved model from file
from keras.models import load_model

model = load_model('mnistCNN.h5')

#----------------------------------------------------#

import coremltools

digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

coreml_model = coremltools.converters.keras.convert(model, input_names = 'image', image_input_names = 'image', output_names = 'digit', class_labels = digit_labels)

coreml_model.author = 'Akhil Waghmare'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Classifies handwritten digit'

coreml_model.input_description['image'] = 'Image of handwriting'
coreml_model.output_description['digit'] = 'Digit classification'

coreml_model.save('MNIST.mlmodel')