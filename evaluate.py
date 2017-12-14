#coding:utf-8
import tensorflow as tf
import cifar10
import get_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def getnext(start,end,input_images):

    temp_images=[]
    for i in range(start,end):
        temp_images.append(input_images[i])
    return temp_images

def get_max(arr):
    max=arr[0]
    label=0
    for i in range(0,len(arr)):
        if arr[i]>max:
            max=arr[i]
            label=i
    return label
def compute_accuracy(result_label,input_label):
    if(len(result_label)!=len(input_label)):
        print u"输入输出label长度不一致"
        return
    count=0.0
    for i in range(len(input_label)):
        if(result_label[i]==input_label[i]):
            count+=1
    return count/len(labels_input)






#用tf处理的数值类型最好都是numpy.float32，有些数值处理器来容易产生异常


def prediction(images_input):

    images = tf.placeholder("float", shape=[None, 24,24,3])



    #卷积1

    w_conv1=weight_variable([5,5,3,64])
    b_conv1=bias_variable([64])
    h_conv1=tf.nn.relu(tf.nn.conv2d(images, w_conv1,[1,1,1,1],padding='SAME')+ b_conv1)


    # bias = tf.reshape(tf.nn.bias_add(conv, b_conv1), conv.get_shape().as_list())
    # conv1 = tf.nn.relu(bias)

    #池化1，正规化
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1') #16*16*64

    #卷积2
    w_conv2=weight_variable([5,5,64,64])
    b_conv2=bias_variable([64])
    h_conv2= (tf.nn.conv2d(h_norm1, w_conv2, [1, 1, 1, 1], padding='SAME')+b_conv2)


    #池化正规化2
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')#8*8*64



    #全连接层1

    w_fc1 = weight_variable([6 * 6 * 64, 384])
    b_fc1 = bias_variable([384])
    h_pool2_flat=tf.reshape(h_pool2,[-1,6*6*64])#

    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)#输出384

    #全连接2
    w_fc2=weight_variable([384,192])
    b_fc2=bias_variable([192])

    h_fc2=tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)#输出


    weight=weight_variable([192,10])
    bais=bias_variable([10])



    prediction=tf.matmul(h_fc2,weight)+bais





    sess = tf.Session()
    saver=tf.train.Saver()


    saver.restore(sess,cifar10.check_point_path)
    print "laod weights"


    step_length=len(images_input)/10
    start=0
    end=start+step_length
    result_label=[]
    for i in range(0,10):
        temp_images = getnext(start, end, images_input)
        start+=step_length
        end+=step_length
        print "process:"+str(i)
        result=sess.run(prediction,feed_dict={images:temp_images})
        for i in result:
            max=get_max(i)
            result_label.append(max)
    return  result_label

labels_input,images_input,beofre_process=get_data.get_test_set()
#get_data.display(beofre_process)

print len(beofre_process)
print len(labels_input)

result_label=prediction(images_input)
accuracy=compute_accuracy(result_label,labels_input)

print"==============================="
print u"准确率为："+str(accuracy)
print u"predict class"
print result_label
print get_data.get_class(result_label)
print"==============================="
print u"real class"
print labels_input
print get_data.get_class(labels_input)






