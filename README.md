# deep_pyramid_fusion

code for our paper "Learning deep pyramid based representations for parsharpening" submitted at IEEE ACCESS

Paper link: https://arxiv.org/pdf/2102.08423.pdf

If you're using the code, then cite our papers,

1) Adeel H, Tahir J, Riaz MM, Ali SS. Siamese Networks Based Deep Fusion Framework for Multi-Source Satellite Imagery. IEEE Access. 2022 Jan 18;10:8728-37.

2) Adeel H, Ali SS, Riaz MM, Kirmani SA, Qureshi MI, Imtiaz J. Learning Deep Pyramid-based Representations for Pansharpening. Arabian Journal for Science and Engineering. 2022 Aug;47(8):10655-66

- All dependencies are mentioned in requirements.txt. Run "pip install -r requirements.txt" in terminal to install all packages in a single go.

(8-band pansharpening)...

- The training and testing images are available from this link (https://drive.google.com/drive/folders/1ODckCjcO44xBfyUHuamRW94BzKy6AEnp?usp=drive_link)

- Test using pre-trained model:

The pretrained model for 8-band pansharpening is available at ./model-8. Currently the model is jointly trained on worldview-2 and worldview-3 images.

Run testing_final.py and adjust the paths accordingly. Currently first four bands and last four bands are saved in different folders. Images are saved in 11 bit tiff format within the range [0 (2^11-1)]

- Visualization: For RGB display, the band index is 5, 3, 1 respectively, i.e., among the sequence of channels, Red channel is 5th, green is 3rd and blue is 1st.

- Training: For training on your own dataset, arrange the dataset according to the specified folders and run training_final.py



