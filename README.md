# Grain structure recognition from NMR scans using semantic segmentation

**Authors**:
*Yaroslav Plutenko*
*Dmytro Babchuk*
*Natalia Rudenko*

### Customer: 
**The Leibniz Institute for Plant Genetics and Crop Plant Research (IPK)**
One of the world's leading international institutions in the field of plant genetics and crop science.  
They conduct researches into the genetic and molecular basis of the traits important for crop productivity.
![alt text](
blob/thumbnails/ipk.JPG "IPK")


They apply 3D scanning of wheat kernels using Nuclear Magnetic Resonance (NMR).
Manually marking recognizable internal structures in order to calculate the volume ratio of each tissue.


NMR allows us to determine the concentration of certain substances (in our case lipids) based on the magnetic properties of certain elements. 
It is widely applied in  medical diagnostics


![alt text](
blob/thumbnails/nmr.JPG "IPK")



The aim of our project: to develop a model that will automatically recognize on the NMR images such grain structures as: 
* endosperm; 
* embryo;
* aleurone layer.



![alt text](
blob/thumbnails/kernel.JPG "kernel")


### Data:
- We received 2 batches of wheat kernels NMR scans, each containing 180 frames (slices) and the same amount of annotated images.
- abt. 10% of these slices are empty and should be discarded. We can apply data augmentation using free scan processing software (Fiji) and open-source Python code.

![alt text](
blob/thumbnails/data.JPG "data")



Technology used: semantic segmentation.
It refers to the process of linking each pixel in an image to a class label.
And could be described as "image classification at a pixel level".

![alt text](
blob/thumbnails/segmentation.JPG "segmentation")




U-net - a convolutional neural network that was initially developed for biomedical image segmentation. 
Then it found wide application due to the ability to make the most use out of limited data.
![alt text](
blob/thumbnails/unet.JPG "U-Net")


### Achievements so far
Using the ready-to-use open-source vgg-unet model (wrapper over Keras framework) with a minimal set of slices (not augmented)
https://colab.research.google.com/drive/1H0iymWw2u7rGHLKoAjIJa-yQ7qDRlfCp 
![alt text](
blob/thumbnails/first_shot.JPG "achivements")



### Problems:
- moderate test results, 
- hi accuracy reported by the model (>95%), credibility in doubt, most likely due to the  large area of background, needs correction 
- issues with saving and restoring trained model, 
- issues with bulk conversion, 
- also model is designed to work with color RGB images, so the output is colored weird but regions are defined more or less correctly.

### Next steps:
- using Keras to build own U-net like architecture
- working with input data: augmenting with Fiji and **imgaug**, refining scripts for local preprocessing.
- stitching git repository and Colab notebook to work seamlessly
- testing various architectural flavors to achieve better accuracy and performance



**April 4th 2020**
**UCU**


