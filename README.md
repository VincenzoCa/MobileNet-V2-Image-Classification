# Image Classification with MobileNet V2

In this [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) notebook we will be classifying images of different animal species living in Oregon using data from the [Oregon Wildlife Dataset](https://www.kaggle.com/virtualdvid/oregon-wildlife), which consists of a collection of 14,000+ labelled images.

This kind of problem is called multi-class [image classification](https://www.tensorflow.org/tutorials/images/classification). It is multi-class because we will be trying to classify mutliple different species of animals.

Our image classifier finds patterns in images using [MobileNet V2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) (from [TensorFlow Hub](https://tfhub.dev/)), which is a family of neural network architectures for efficient on-device image classification and related tasks. This model can be used with the `hub.KerasLayer` as follows.
```
m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4")
])
m.build([None, 224, 224, 3])  # Batch input shape.
```
This input layer takes in our images and finds patterns in them based on the patterns `mobilenet_v2_130_224` has found.

The input images are expected to have color values in the range [0,1], following the [common image input](https://www.tensorflow.org/hub/common_signatures/images#input) conventions. For this model, the size of the input images is fixed to height x width = **224 x 224 pixels**. For preprocessing our data, we will be using **TensorFlow 2.3**.

This notebook runs a lot faster with **GPU acceleration**. If you wish to do so, please use the Colab menu to change the runtime to GPU via `Runtime` -> `Change runtime type`, and then restart the runtime with `Runtime` -> `Restart runtime`.

![Predictions](https://github.com/VincenzoCa/MobileNet-V2-Image-Classification/blob/master/img/predictions.png)
<p align="center">
Predictions with mobilenet_v2_130_224
</p>   
