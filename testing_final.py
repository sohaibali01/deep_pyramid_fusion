
import os
import skimage
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

import model_final

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()
num_bands = 8  # number of input's channels 
img_size = 512 # assuming square image size

if num_bands == 8:
   model_path = './model-8/'
   pan_path = './TestData/pan-8/' 
   mul_path1 = './TestData/mul-8/bands1-4/' 
   mul_path2 = './TestData/mul-8/bands5-8/' 
   results_path1 = './TestData/results-8/bands1-4/' # output path of first 4 bands 
   results_path2 = './TestData/results-8/bands5-8/' # output path of last 4 bands

else:
   model_path = './model-4/'
   pan_path = './TestData/pan-4/' # the path of pan images
   mul_path1 = './TestData/mul-4/' # the path of multispectral images
   results_path1 = './TestData/results-4/' # output path of first 4 bands 

def readTIFImage(imgpath):
    image_decoded = skimage.io.imread(imgpath.decode("utf-8"))
    image_array = np.array(image_decoded,dtype=np.uint16)
    return image_array

def _parse_function_4Bands(panfilename,mulfilename1): 
      
  image_pan = tf.py_func(readTIFImage, [panfilename], tf.uint16)
  pan = tf.cast(image_pan[:,:,None], tf.float32)/((2.0**16)-1)  

  mul1 = tf.py_func(readTIFImage, [mulfilename1], tf.uint16)
  mulF = tf.cast(mul1, tf.float32)/((2.0**16)-1)

  offset_height, offset_width = 0, 0     
  panImage = tf.image.crop_to_bounding_box(pan, offset_height, offset_width, img_size, img_size)
  mulImage = tf.image.crop_to_bounding_box(mulF, offset_height, offset_width, int(img_size/4), int(img_size/4))

  return panImage, mulImage 

def _parse_function_8Bands(panfilename,mulfilename1, mulfilename2 ):   

  image_pan = tf.py_func(readTIFImage, [panfilename], tf.uint16)
  pan = tf.cast(image_pan[:,:,None], tf.float32)/((2.0**16)-1)  

  mul1 = tf.py_func(readTIFImage, [mulfilename1], tf.uint16)
  mul1F = tf.cast(mul1, tf.float32)/((2.0**16)-1)  

  mul2 = tf.py_func(readTIFImage, [mulfilename2], tf.uint16)
  mul2F = tf.cast(mul2, tf.float32)/((2.0**16)-1)  
  mulF=tf.concat([mul1F, mul2F], tf.rank(mul2F)-1)
 
  offset_height, offset_width = 0, 0     
  panImage = tf.image.crop_to_bounding_box(pan, offset_height, offset_width, img_size, img_size)
  mulImage = tf.image.crop_to_bounding_box(mulF, offset_height, offset_width, int(img_size/4), int(img_size/4))
  
  return panImage, mulImage 


if __name__ == '__main__':
   panName = os.listdir(pan_path)
   mulName = os.listdir(mul_path1)
   num_img = len(panName)

   whole_pathpan = []
   whole_pathmul = []
   for i in range(num_img):
      whole_pathpan.append(pan_path + panName[i])
      whole_pathmul.append(mul_path1 + mulName[i])
      
   if num_bands == 8:
      whole_pathmul2 = []
      mulName2 = os.listdir(mul_path2)
      for i in range(num_img):
         whole_pathmul2.append(mul_path2 + mulName2[i])
        
   panfilename_tensor = tf.convert_to_tensor(whole_pathpan, dtype=tf.string)  
   mulfilename_tensor = tf.convert_to_tensor(whole_pathmul, dtype=tf.string) 

   if num_bands == 8:
        mulfilename_tensor2 = tf.convert_to_tensor(whole_pathmul2, dtype=tf.string)    
        dataset = tf.data.Dataset.from_tensor_slices((panfilename_tensor, mulfilename_tensor, mulfilename_tensor2))      
        dataset = dataset.map(_parse_function_8Bands)
   else:
        dataset = tf.data.Dataset.from_tensor_slices((panfilename_tensor, mulfilename_tensor))
        dataset = dataset.map(_parse_function_4Bands) 
      
   dataset = dataset.prefetch(buffer_size=10)
   dataset = dataset.batch(batch_size=1).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   pan, mul  = iterator.get_next()  

   panpyramid = model_final.panInference(pan)
   mulpyramid = model_final.mulInference(mul)

   out = model_final.reconstructFromPyramids(panpyramid,mulpyramid, int(img_size/4), 1)
   output = tf.clip_by_value(out[2], 0., 1.)
   output = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True   
   saver = tf.train.Saver()
   
   with tf.Session(config=config) as sess:
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):  
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model 
              saver.restore(sess, ckpt)
              print ("Loading model")

          for i in range(num_img):     
             fused = sess.run(output)            
             fused = np.uint16(fused* ((2.0**16)-1))
             index = panName[i].rfind('.')
             name = panName[i][:index]
             skimage.io.imsave(results_path1 + name +'.TIF', fused[:,:,0:4]) 
             if num_bands == 8: 
                skimage.io.imsave(results_path2 + name +'.TIF', fused[:,:,4:8]) 
             print('%d / %d images processed' % (i+1,num_img))
              
      print('All done')
   sess.close()   
   
   #plt.subplot(1,3,1)     
   #plt.imshow(ori1[0,:,:,:].squeeze())          
   #plt.title('pan')
   #plt.subplot(1,3,2)     
   #plt.imshow(upSampled[:,:,0:3])          
   #plt.title('mulUpSampled')
   #plt.subplot(1,3,3)    
   #plt.imshow(fused[:,:,0:3])
   #plt.title('fused')
   #plt.show()      