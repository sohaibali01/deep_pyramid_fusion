# deep_pyramid_fusion

code for our paper "Learning deep pyramid based representations for parsharpening" submitted at IEEE TIP

The training and testing imaages a available from this link

- Test using pre-trained model:
The pretrained model is available at ./model-8
Run testing_final.py and adjust the paths accordingly. Currently first four bands and last four bands are saved in different folders. Images are saved in 11 bit tiff format within the range [0 (2^11-1)]

- Visualization: For RGB display, the band index is 5, 3, 1 respectively, i.e., among the sequence of channels, Red channel is 5th, green is 3rd and blue is 1st.

- Training: For training on your own dataset, arrange the dataset according to the specified folders and run training_final.py
