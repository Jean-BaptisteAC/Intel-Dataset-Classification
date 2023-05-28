# Intel Classification Dataset
In this project, we will tackle a classification problem using advanced Convolutional Neural Networks (CNN) and the Intel Image dataset, featuring 6 classes of color pictures in 150x150 pixels resolution.

## Using the code
You can use the jupyter version of the project with the file ```Advanced CNN.ipynb```.
The dataset used in the code can be fond on the kaggle website: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

## Importing data

This part of the codes handles the data loading from the specific folder in the user's computer. One need to change the path at line 24 to be able to charge the data propely.
The data is retrived from the train dataset and test dataset in order to provide testing at the end of the training process.
For the rest of the project, we will be using tensorflow as the main library for neural network creation.

```
import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                               
from tqdm import tqdm

IMAGE_SIZE = (150, 150)
# Chargement des données d'entraînement
def load_data():
    datasets = ['C:\\Users\\etern\\Documents\\PROJET_CNN_JB_1\\seg_train\\seg_train', 'C:\\Users\\etern\\Documents\\PROJET_CNN_JB_1\\seg_test\\seg_test']

    output = []
    #
    class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    nb_classes = len(class_names)

#TRANSFORMATION image matric + label en vector one-hot
    for dataset in datasets:
        images = []
        labels = []
        print("Loading {}".format(dataset))

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)
                images.append(image/255)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        labels = tf.keras.utils.to_categorical(labels, nb_classes)  # Convert labels to one-hot vectors

        output.append((images, labels))

    return output
    
(train_images, train_labels), (test_images, test_labels) = load_data()
```

## Modeling

For the creation of our CNN, we used different layers of convolutions, maxpooling and dense layer for class prediction. We inspired our self from the VGG-architecture: Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/abs/1409.1556).

The important points from this architecture are:
- The great number of convolution stacks, comprising convolutions layers with a small kernel
- A stride of 1 with padding set to 'same', in order to keep the same resolution between convolution layers.
- Maxpooling layers with a size of (2,2)

For compilation of our model, we used the standard Adam optimizer, the categorical crossentropy loss function which is great for classification, and the accuracy metric for evaluation of our model. 

The summary of the model is then displayed thanks to the tensorflow method in order to visualize our achitecture.
We have in total, around half a million parameters in our model.

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 150, 150, 32)      896       
                                                                 
 conv2d_1 (Conv2D)           (None, 150, 150, 32)      9248      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 75, 75, 32)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 75, 75, 64)        18496     
                                                                 
 conv2d_3 (Conv2D)           (None, 75, 75, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 37, 37, 64)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 37, 37, 128)       73856     
                                                                 
 conv2d_5 (Conv2D)           (None, 37, 37, 128)       147584    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 18, 18, 128)      0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 18, 18, 64)        73792     
                                                                 
 conv2d_7 (Conv2D)           (None, 18, 18, 64)        36928     
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 9, 9, 64)         0         
 2D)                                                             
                                                                 
 reshape (Reshape)           (None, 5184)              0         
                                                                 
 dense (Dense)               (None, 16)                82960     
                                                                 
 dense_1 (Dense)             (None, 6)                 102       
                                                                 
=================================================================
Total params: 480,790
Trainable params: 480,790
Non-trainable params: 0
_________________________________________________________________
````

## Results

At the end of the training, we achieved around 96% accuracy on the train set, and around 0.81% accuracy on the test set, which is a good result for our model. 
The time needed for total computation is rather small with around 1h for 10 epochs (our computer has 128Go RAM with a strong CPU, and we didn't use GPU acceleration).

```
...
Epoch 10/10
878/878 [==============================] - 429s 489ms/step - loss: 0.1132 - accuracy: 0.9608 - val_loss: 0.7838 - val_accuracy: 0.8070
```

When looking at the confusion matrix below, we can see that there is no perticular class imbalance in our predictions. The most errors occur between the classes "Glacier" and "Mountain", which is expected as a mountain covered in snow is very close to a glacier, and between the classes "Street" and "Building" due to the similarity of artificial landscapes.

![image](https://github.com/Jean-BaptisteAC/Intel-Dataset-Classification/assets/66775006/8ad4f74c-6f44-4f5e-afdf-2894eda92c95)



## Conclusion

The intel dataset is a rather heavy dataset to work on, and needed the implementation of a very deep CNN with around 14 layers in order to yield good results.
This project was fun to make because the pictures used are highly detailed and could be pictures from one personal photos.
