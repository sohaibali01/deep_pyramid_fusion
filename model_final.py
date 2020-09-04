
# author: Sohaib Ali Syed (sohaibali01@gmail.com)

import numpy as np
import tensorflow as tf
import skimage.io

num_pyramids = 3  # number of pyramid levels
num_blocks = 3    # number of blocks
num_bands = 8  # number of input's channels 

def angleLoss(images, refImages): 
    images = tf.clip_by_value(images, 1e-6, 1.)
    refImages = tf.clip_by_value(refImages, 1e-6, 1.) 
    imagesNormL2=tf.math.sqrt(tf.reduce_sum(tf.math.square(images),3))
    normImages=tf.math.divide(images,imagesNormL2[:,:,:,None])
    refImagesNormL2=tf.math.sqrt(tf.reduce_sum(tf.math.square(refImages),3))
    normRefImages=tf.math.divide(refImages,refImagesNormL2[:,:,:,None])
    spectralAngle = tf.reduce_sum(normImages*normRefImages,3)
    lossSpectralAngle = 1.-tf.reduce_mean(spectralAngle)
    return lossSpectralAngle

# leaky ReLU
def lrelu(x, leak = 0.2, name = 'lrelu'):   
    with tf.variable_scope(name):
         return tf.maximum(x, leak*x, name = name)   

######## Laplacian  Pyramid ########
def lap_split(img,kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1,2,2,1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(low, kernel*4, tf.shape(img), [1,2,2,1])
        high = img - low_upsample
    return low, high

def LaplacianPyramid(img,kernel,n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]

# create kernel
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables



# sub network
def subnet(images, residual,  num_feature, num_input_channels, num_output_channels):

    kernel0 = create_kernel(name='weights_0', shape=[5, 5, num_input_channels, num_feature])
    biases0 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_0')  
  
    kernel1=create_kernel(name='weights_1', shape=[5, 5, num_feature, num_feature])
    biases1=tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_1')  

    kernel2=create_kernel(name='weights_2', shape=[5, 5, num_feature, num_feature])
    biases2=tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_2') 

    kernel3=create_kernel(name='weights_3', shape=[1, 1, num_feature*3, num_feature])
    biases3=tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_3')  

    kernel4 = create_kernel(name='weights_4', shape=[5, 5, num_feature, num_output_channels])
    biases4 = tf.Variable(tf.constant(0.0, shape=[num_output_channels], dtype=tf.float32), trainable=True, name='biases_4')  

  #  1st layer
    with tf.variable_scope('1st_layer'):    
         conv0 = tf.nn.conv2d(images, kernel0, [1, 1, 1, 1], padding='SAME')
         bias0 = tf.nn.bias_add(conv0, biases0) 
         bias0 = lrelu(bias0) # leaky ReLU

         out_block =  bias0
         concatTensor = out_block
  #  recursive blocks
    for i in range(num_blocks):
        with tf.variable_scope('block_%s'%(i+1)):
         
             conv1 = tf.nn.conv2d(out_block, kernel1, [1, 1, 1, 1], padding='SAME')
             bias1 = tf.nn.bias_add(conv1, biases1) 
             bias1 = lrelu(bias1) 
  
             conv2 = tf.nn.conv2d(bias1, kernel2, [1, 1, 1, 1], padding='SAME')
             bias2 = tf.nn.bias_add(conv2, biases2) 
             bias2 = lrelu(bias2) 
  
             inner=tf.concat([out_block, bias1, bias2], 3)
             conv3 = tf.nn.conv2d(inner, kernel3, [1, 1, 1, 1], padding='SAME')
             bias3 = tf.nn.bias_add(conv3, biases3) 
             bias3 = lrelu(bias3) 

             out_block = tf.add(bias3, bias0) #  shortcut
             concatTensor= tf.concat([concatTensor, out_block], 3)
  
    #  concatenation layer
    with tf.variable_scope('concatenation'): 
         kernel5 = create_kernel(name='weights_5', shape=[1, 1, num_feature*(num_blocks+1), num_feature])
         biases5 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_5')  
         conv5 = tf.nn.conv2d(concatTensor, kernel5, [1, 1, 1, 1], padding='SAME')
         bias5 = tf.nn.bias_add(conv5, biases5) 
         bias5 = lrelu(bias5) 

   #  reconstruction layer
    with tf.variable_scope('recons'):    
         conv = tf.nn.conv2d(bias5, kernel4, [1, 1, 1, 1], padding='SAME')
         recons = tf.nn.bias_add(conv, biases4) 
         recons = lrelu(recons) 
         final_out = tf.add(recons, residual) #  shortcut
  
    return final_out

def reconstructFromPyramids(pan_pyramid, mul_pyramid, image_size, batch_size):
    num_feature = 48
    k = np.float32([.0625, .25, .375, .25, .0625]) # Gaussian kernel for image pyramid
    k = np.outer(k, k) 
    kernel = k[:,:,None,None]/k.sum()*np.eye(num_bands, dtype=np.float32) 

    with tf.variable_scope('outputNetwork1'):  
         #out1 = mul_pyramid[0]+pan_pyramid[0]
         sh=[batch_size, image_size*2, image_size*2, num_bands] 
         outShape= tf.convert_to_tensor(sh,dtype=np.int32)
         #inp = tf.concat([pan_pyramid[0], mul_pyramid[0]], tf.rank(pan_pyramid[0])-1)
         #LF = subnet_reconstruction(inp, int(num_feature/4), num_bands+1, num_bands)
         LFU = tf.nn.conv2d_transpose(mul_pyramid[0], kernel*4, outShape, [1,2,2,1])  

    with tf.variable_scope('outputNetwork2', reuse=tf.AUTO_REUSE):  
         sh2=[batch_size, image_size*4, image_size*4, num_bands] 
         outShape2= tf.convert_to_tensor(sh2,dtype=np.int32)
         inp1 = tf.concat([pan_pyramid[1],  LFU], tf.rank(pan_pyramid[1])-1)
         LF1 = subnet(inp1, LFU, int(num_feature), (num_bands)+1, num_bands) 
         LF1U = tf.nn.conv2d_transpose(LF1, kernel*4, outShape2, [1,2,2,1])

    with tf.variable_scope('outputNetwork2', reuse=tf.AUTO_REUSE):  
         #finalImage = subnet(mul_pyramid[2]+pan_pyramid[2]+LF1U, int(num_feature), num_bands)
         inp2 = tf.concat([pan_pyramid[2], LF1U], tf.rank(pan_pyramid[2])-1)
         finalImage = subnet(inp2, LF1U, int(num_feature), (num_bands)+1, num_bands) 

    #with tf.variable_scope('outputNetwork',, reuse = tf.AUTO_REUSE):  
         #final = subnet(out3, int(num_feature), num_bands)
            
    out_pyramid = []        
    out_pyramid.append(mul_pyramid[0])
    out_pyramid.append(LF1)
    out_pyramid.append(finalImage)

    return out_pyramid

# LPNet structure
def panInference(images):
    with tf.variable_scope('paninference'):
         k = np.float32([.0625, .25, .375, .25, .0625]) # Gaussian kernel for image pyramid
         k = np.outer(k, k) 
         kernel = k[:,:,None,None]/k.sum()*np.eye(1, dtype=np.float32)
         pyramid = LaplacianPyramid(images, kernel, (num_pyramids - 1)) 
            
         outout_pyramid = []        
         outout_pyramid.append(pyramid[0])
         outout_pyramid.append(pyramid[1])
         outout_pyramid.append(pyramid[2])
       
         return outout_pyramid

def mulInference(images):
    with tf.variable_scope('mulinference'):
         outout_pyramid = []        
         outout_pyramid.append(images)
        
    return outout_pyramid


if __name__ == '__main__':
    tf.reset_default_graph()   
    #tf.enable_eager_execution()

    #pan_path = './TrainData/mul-8/bands1-4/401.TIF' # rainy images
    #pan_file = tf.convert_to_tensor(pan_path, dtype=tf.string)
    
    ##pan_file = tf.read_file(pan_file) 
    #pan_path2=pan_file.numpy()

    #print(pan_path2)
    ##image_string = tf.read_file(pan_path)
    #testI = skimage.io.imread(pan_path2.decode("utf-8"))
    #print(testI.dtype)
    #print(testI.shape)
    #testI = np.array(testI,dtype=np.uint16)
    #print(np.amax(testI))
    #pan = tf.cast(testI, tf.float32)/(2**16)

    mulInput = tf.placeholder(tf.float32, [20,32,32,num_bands])
    panInput = tf.placeholder(tf.float32, [20,128,128,1])

    #mulUpSampled  = upSampleNet(mulInput,32,10)
    panpyramid = panInference(panInput)
    mulpyramid = mulInference(mulInput)
    #output = tf.clip_by_value(pyramid[-1], 0., 1.)
    output = reconstructFromPyramids(panpyramid, mulpyramid, 32, 20)

    var_list = tf.trainable_variables()   
    print("Total parameters' number: %d" 
         %(np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))  
