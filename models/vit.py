"""
Title: Image classification with Vision Transformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/18
Last modified: 2021/01/18
Description: Implementing the Vision Transformer (ViT) model for image classification.
Accelerator: GPU
"""

"""
## Introduction

This example implements the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
model by Alexey Dosovitskiy et al. for image classification,
and demonstrates it on the CIFAR-100 dataset.
The ViT model applies the Transformer architecture with self-attention to sequences of
image patches, without using convolution layers.

"""

"""
## Imports
"""

import time
import matplotlib.pyplot as plt

#os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import keras
from keras import layers
from keras import ops
from sklearn.base import BaseEstimator
import cv2

"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        #x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


"""
Let's display patches for a sample image
"""

def visualise_patches(x_train):
    plt.figure(figsize=(4, 4))
    image = 255*x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype("uint8"), cmap='gray')
    plt.axis("off")

    resized_image = ops.image.resize(
        ops.convert_to_tensor([image]), size=(IMAGE_SIZE[0], IMAGE_SIZE[1])
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {IMAGE_SIZE[0]} X {IMAGE_SIZE[1]}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = ops.reshape(patch, (patch_size, patch_size, 1))
        plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"), cmap='gray')
        plt.axis("off")


"""
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


"""
## Build the ViT model

The ViT model consists of multiple Transformer blocks,
which use the `layers.MultiHeadAttention` layer as a self-attention mechanism
applied to the sequence of patches. The Transformer blocks produce a
`[batch_size, num_patches, projection_dim]` tensor, which is processed via an
classifier head with softmax to produce the final class probabilities output.

Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
which prepends a learnable embedding to the sequence of encoded patches to serve
as the image representation, all the outputs of the final Transformer block are
reshaped with `layers.Flatten()` and used as the image
representation input to the classifier head.
Note that the `layers.GlobalAveragePooling1D` layer
could also be used instead to aggregate the outputs of the Transformer block,
especially when the number of patches and the projection dimensions are large.
"""


def create_vit_classifier(
        input_shape,
        projection_dim,
        patch_size,
        transformer_layers,
        num_heads,
        mlp_head_unit_1,
        mlp_head_unit_2,
    ):
    
    mlp_head_units = [
        mlp_head_unit_1,
        mlp_head_unit_2,
    ]  # Size of the dense layers of the final classifier

    num_patches = (input_shape[0] // patch_size) ** 2
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    
    inputs = keras.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, #dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    #representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units)
    # Classify outputs.
    logits = layers.Dense(1, activation='sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training; {} : {}".format(epoch, logs.keys(), logs.values()), end='\r')

"""
## Compile, train, and evaluate the mode
"""


def model_train(
        model,
        learning_rate,
        weight_decay,
        batch_size,
        num_epochs,
        x_train,
        y_train
    ):

    
    checkpoint_filepath = './tmp.weights.h5'
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy'
        ],
    )
    
    '''
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    '''

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.,
        #callbacks=[checkpoint_callback],
        callbacks=[CustomCallback()],
        verbose=0,
    )
 
    return history


def model_test(model, checkpoint_filepath, x_test, y_test):
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")


def plot_history(history, item):
    plt.plot(history.history[item], label=item)
    #plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig('ViT train history.eps')
    plt.savefig('ViT train history.png')
    #plt.show()


"""
After 100 epochs, the ViT model achieves around 55% accuracy and
82% top-5 accuracy on the test data. These are not competitive results on the CIFAR-100 dataset,
as a ResNet50V2 trained from scratch on the same data can achieve 67% accuracy.

Note that the state of the art results reported in the
[paper](https://arxiv.org/abs/2010.11929) are achieved by pre-training the ViT model using
the JFT-300M dataset, then fine-tuning it on the target dataset. To improve the model quality
without pre-training, you can try to train the model for more epochs, use a larger number of
Transformer layers, resize the input images, change the patch size, or increase the projection dimensions.
Besides, as mentioned in the paper, the quality of the model is affected not only by architecture choices,
but also by parameters such as the learning rate schedule, optimizer, weight decay, etc.
In practice, it's recommended to fine-tune a ViT model
that was pre-trained using a large, high-resolution dataset.
"""

class ViTWrapper(BaseEstimator):
    def __init__(
            self,
            projection_dim=64,
            patch_size=6,
            transformer_layers=3,
            num_heads=4,
            mlp_head_unit_1=64,
            mlp_head_unit_2=32
        ):
        
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_head_unit_1 = mlp_head_unit_1
        self.mlp_head_unit_2 = mlp_head_unit_2
        
    
    def get_params(self, deep=True):
        return {
            'projection_dim': self.projection_dim,
            'patch_size': self.patch_size,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'mlp_head_unit_1': self.mlp_head_unit_1,
            'mlp_head_unit_2': self.mlp_head_unit_2,
        }
    
    
    def create(self):
        input_shape = (50, 50, 1)
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 250
        self.num_epochs = 100  # For real training, use num_epochs=100. 10 is a test value
        self.model = create_vit_classifier(
            input_shape,
            self.projection_dim,
            self.patch_size,
            self.transformer_layers,
            self.num_heads,
            self.mlp_head_unit_1,
            self.mlp_head_unit_2,
        )
        return self
    
    
    def fit(self, inputs, targets):
        input_shape = (50, 50, 1)
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 250
        self.num_epochs = 100  # For real training, use num_epochs=100. 10 is a test value
        self.model = create_vit_classifier(
            input_shape,
            self.projection_dim,
            self.patch_size,
            self.transformer_layers,
            self.num_heads,
            self.mlp_head_unit_1,
            self.mlp_head_unit_2,
        )
        history = model_train(
            self.model,
            self.learning_rate,
            self.weight_decay,
            self.batch_size,
            self.num_epochs,
            inputs,
            targets,
        )
        return self
        
        
    def score(self, inputs, targets):
        _, accuracy = self.model.evaluate(inputs, targets)
        return accuracy
