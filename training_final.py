
# author: Sohaib Ali Syed (sohaibali01@gmail.com)

#import plaidml.keras
#plaidml.keras.install_backend()

import os
import re
import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import skimage
import random

from model_final import panInference, mulInference, LaplacianPyramid, upSampleNet, angleLoss,reconstructFromPyramids 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

num_bands = 8
num_pyramids = 3       # number of pyramid levels
learning_rate = 1e-3   # learning rate
iterations = int(2e5)  # iterations
batch_size = 10        # batch size
patch_size = 48       # low resolution patch size 
repeat_factor = 5
model_name = 'model-epoch'    # name of saved model

if num_bands == 8:
    save_model_path = './model-8/'  # path of saved model
    pan_path = './TrainData/pan-8/' # panchromatic images
    mul_path1 = './TrainData/mul-8/bands1-4/' # first 4 bands of low resolution multispectral images
    mul_path2 = './TrainData/mul-8/bands5-8/' # last 4 bands of low resolution multispectral images
    gt_path1 = './TrainData/label-8/bands1-4/'    # ground truth
    gt_path2 = './TrainData/label-8/bands5-8/'    # ground truth

else:
    save_model_path = './model-4/'  # path of saved model
    pan_path = './TrainData/pan-4/' # panchromatic images
    mul_path1 = './TrainData/mul-4/' # first 4 bands of low resolution multispectral images
    gt_path1 = './TrainData/label-4/'    # ground truth

def readResizeTIFImage(imgpath):
    image_decoded = skimage.io.imread(imgpath.decode("utf-8"))
    imageN= np.array(image_decoded,dtype=np.uint16)
    #image_decoded = skimage.io.imread(imgpath)
    image_array = np.array(imageN, dtype=np.float32) / ((2.0**16)-1)
    image_resized = skimage.transform.resize(image_array, (image_array.shape[0] * 4, image_array.shape[1] * 4))
    image_array2 = np.array(image_resized, dtype=np.float32) 
    return image_array2

def readTIFImage(imgpath):
    image_decoded = skimage.io.imread(imgpath.decode("utf-8"))
    image_array = np.array(image_decoded,dtype=np.uint16)
    return image_array

def _parse_function_4Bands(pan_path, mul_path, gt_path, patch_size = patch_size):
 
    image_pan = tf.py_func(readTIFImage, [pan_path], tf.uint16)
    pan = tf.cast(image_pan[:,:,None], tf.float32)/((2.0**16)-1)

    mul = tf.py_func(readTIFImage, [mul_path], tf.uint16)
    mul32 = tf.cast(mul, tf.float32)/((2.0**16)-1)

    label = tf.py_func(readTIFImage, [gt_path], tf.uint16)
    label32 = tf.cast(label, tf.float32)/((2.0**16)-1)

    maxInd=128-patch_size
    offset_height, offset_width = random.randint(1, maxInd-2), random.randint(1, maxInd-2)      

    panImage = tf.image.crop_to_bounding_box(pan, offset_height*4, offset_width*4, patch_size*4, patch_size*4)
    mulImage = tf.image.crop_to_bounding_box(mul32, offset_height, offset_width, patch_size, patch_size)
    Label = tf.image.crop_to_bounding_box(label32, offset_height*4, offset_width*4, patch_size*4, patch_size*4)
    
    return panImage, mulImage, Label 

def _parse_function_8Bands(pan_path, mul_path1, mul_path2, gt_path1, gt_path2, patch_size = patch_size):  
 
    image_pan = tf.py_func(readTIFImage, [pan_path], tf.uint16)
    pan = tf.cast(image_pan[:,:,None], tf.float32)/((2.0**16)-1)
    
    mul1 = tf.py_func(readTIFImage, [mul_path1], tf.uint16)
    mul1F = tf.cast(mul1, tf.float32)/((2.0**16)-1)
    mul2 = tf.py_func(readTIFImage, [mul_path2], tf.uint16)
    mul2F = tf.cast(mul2, tf.float32)/((2.0**16)-1)
    mulF=tf.concat([mul1F, mul2F], tf.rank(mul2)-1)
    
    gt1 = tf.py_func(readTIFImage, [gt_path1], tf.uint16)
    gt1F = tf.cast(gt1, tf.float32)/((2.0**16)-1)
    gt2 = tf.py_func(readTIFImage, [gt_path2], tf.uint16)
    gt2F = tf.cast(gt2, tf.float32)/((2.0**16)-1)
    labelF=tf.concat([gt1F, gt2F], tf.rank(gt2F)-1)
    
    maxInd=128-patch_size
    offset_height, offset_width = random.randint(1, maxInd-2), random.randint(1, maxInd-2)      

    panImage = tf.image.crop_to_bounding_box(pan, offset_height*4, offset_width*4, patch_size*4, patch_size*4)
    mulImage = tf.image.crop_to_bounding_box(mulF, offset_height, offset_width, patch_size, patch_size)
    Label = tf.image.crop_to_bounding_box(labelF, offset_height*4, offset_width*4, patch_size*4, patch_size*4)
    
    return panImage, mulImage, Label 

if __name__ == '__main__':  

    tf.reset_default_graph()
    #tf.enable_eager_execution()

    if tf.test.gpu_device_name():
        print(tf.test.gpu_device_name())

    pan_files = os.listdir(pan_path)
    for i in range(len(pan_files)):
        pan_files[i] = pan_path + pan_files[i]
    
    pan_files = pan_files * repeat_factor
    totalFiles = len(pan_files)
    epochSize=int(float(totalFiles)/float(batch_size))

    mul_files1 = os.listdir(mul_path1)
    for i in range(len(mul_files1)):
        mul_files1[i] = mul_path1 + mul_files1[i]
    mul_files1 = mul_files1 * repeat_factor

    label_files1 = os.listdir(gt_path1)       
    for i in range(len(label_files1)):
        label_files1[i] = gt_path1 + label_files1[i] 
    label_files1 = label_files1 * repeat_factor

    if num_bands == 8:
        mul_files2 = os.listdir(mul_path2)
        for i in range(len(mul_files2)):
            mul_files2[i] = mul_path2 + mul_files2[i]
        mul_files2 = mul_files2 * repeat_factor

        label_files2 = os.listdir(gt_path1)       
        for i in range(len(label_files2)):
            label_files2[i] = gt_path2 + label_files2[i]     
        label_files2 = label_files2 * repeat_factor
    
    pan_files = tf.convert_to_tensor(pan_files, dtype=tf.string)
    mul_files1 = tf.convert_to_tensor(mul_files1, dtype=tf.string)    
    label_files1 = tf.convert_to_tensor(label_files1, dtype=tf.string)  
    if num_bands == 8:
        mul_files2 = tf.convert_to_tensor(mul_files2, dtype=tf.string)    
        label_files2 = tf.convert_to_tensor(label_files2, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((pan_files, mul_files1, mul_files2, label_files1, label_files2))      
    else:
        dataset = tf.data.Dataset.from_tensor_slices((pan_files, mul_files1, label_files1))
         
    dataset = dataset.shuffle(buffer_size= 5 * totalFiles)
    if num_bands == 8:
        dataset = dataset.map(_parse_function_8Bands)
    else:
        dataset = dataset.map(_parse_function_4Bands)

    dataset = dataset.prefetch(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size,drop_remainder=True).repeat()  
    iterator = dataset.make_one_shot_iterator()   
    panImages, mulImages, labels = iterator.get_next()  

    k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
    k = np.outer(k, k) 
    kernel = k[:,:,None,None]/k.sum()*np.eye(num_bands, dtype = np.float32)
    labels_LP = LaplacianPyramid(labels, kernel, (num_pyramids - 1)) # labels Laplacian pyramid
    
    labels1 = labels_LP[1] + tf.nn.conv2d_transpose(labels_LP[0], kernel*4, tf.shape(labels_LP[1]), [1,2,2,1])  
    pan_pyramid = panInference(panImages) # 
    mul_pyramid = mulInference(mulImages) # 
    outImages = reconstructFromPyramids(pan_pyramid, mul_pyramid, patch_size, batch_size)

    HF1lossL1 = tf.norm(outImages[1] - labels1)    # L1 loss
    HF2lossL1 = tf.norm(outImages[2]- labels)    # L1 loss

    loss = HF1lossL1 + HF2lossL1  
    g_optim =  tf.train.AdamOptimizer(learning_rate).minimize(loss) # Optimization method: Adam
    
    all_vars = tf.trainable_variables()  
    saver = tf.train.Saver(var_list=all_vars, max_to_keep = 5)  

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
      
       sess.run(tf.group(tf.global_variables_initializer(), 
                         tf.local_variables_initializer()))
       tf.get_default_graph().finalize()	
              
       if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
          ckpt = tf.train.latest_checkpoint(save_model_path)
          saver.restore(sess, ckpt)  
          ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
          start_point = int(ckpt_num[0]) + 1     
          print("loaded successfully")
       else:  # re-training when no models found
          print("re-training")
          start_point = 0  
         
       #check_pan,check_mul, check_label =  sess.run([panImages, mulImagesUpsampled ,labels])
       #print("check patch pair:")  
       #plt.subplot(1,3,1)     
       #plt.imshow(check_pan[4,:,:,:].squeeze())
       #plt.title('pan')  
       #plt.subplot(1,3,2)     
       #plt.imshow(check_mul[4,:,:,0:3])
       #plt.title('mul')         
       #plt.subplot(1,3,3)    
       #plt.imshow(check_label[4,:,:,0:3])
       #plt.title('ground truth')      
       #plt.show()    
     
       start = time.time()    
       avgTrainingLoss=0
       minAvgLoss=1e10
       for j in range(start_point,iterations):                           
           _, Training_Loss, HF1lossL1O, HF2lossL1O,   = sess.run([g_optim,loss, HF1lossL1, HF2lossL1])  # training
           avgTrainingLoss = avgTrainingLoss + Training_Loss
           if ( np.isnan(Training_Loss)  or np.isnan(HF1lossL1O) or np.isnan(HF2lossL1O)  ):
                        print("nan aappears.... returning... please re run from last checkpoint")
                        break
           if np.mod(j+1,epochSize) == 0 : # print each epoch
                  print ('%d / %d iteraions, Average: Training Loss  = %.4f ' 
                    % (j+1, iterations, avgTrainingLoss/epochSize, ))           
                  if j > 1000 and avgTrainingLoss < minAvgLoss:
                        save_path_full = os.path.join(save_model_path, model_name) 
                        saver.save(sess, save_path_full, global_step = j+1) # save model whenever minimal avg loss
                        print ('model saved')
                        minAvgLoss = avgTrainingLoss  

                  avgTrainingLoss=0
             
           #if np.mod(j+1,100) == 0 and j != 0:    
           #   end = time.time() 
           #   print ('%d / %d iteraions, Currentt Batch: Training Loss  = %.4f, upsampleLossL1  = %.4f, upsampleLossSSIM  = %.4f, upsampleLossAngle  = %.4f, LFlossL1  = %.4f,LFlossAngle = %.4f,HF1lossL1  = %.4f,HF1lossSSIM  = %.4f,HF2lossL1  = %.4f,HF2lossSSIM  = %.4f, OutlossL1  = %.4f, OutlossSSIM  = %.4f, OutlossAngle  = %.4f, running time = %.1f s' 
           #          % (j+1, iterations, Training_Loss,upsampleLossL1O,upsampleLossSSIMO, upsampleLossAngleO, LFlossL1O, LFlossAngleO, HF1lossL1O, HF1lossSSIMO, HF2lossL1O, HF2lossSSIMO , OutlossL1O , OutlossSSIMO , OutlossAngleO, (end-start)))          
           #   start = time.time()  
              
       print ('training finished') 
    sess.close()