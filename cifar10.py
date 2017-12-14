#coding:utf-8
import tensorflow as tf
import time
import get_data
import os
data_path="/Users/zx142489/workspace/CNN_cifar10/cifar-10-batches-bin/test_batch.bin"
check_point_path="/Users/zx142489/workspace/CNN_cifar10/cifar-10-batches-bin/cifar10.ckpt"

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def getnext(start,end,input_images,input_labels):
    temp_labels=[]
    temp_images=[]
    for i in range(start,end):
        temp_labels.append(input_labels[i])
        temp_images.append(input_images[i])
    return temp_images,temp_labels






def start_train():
    varable_list=[]
    #用tf处理的数值类型最好都是numpy.float32，有些数值处理器来容易产生异常
    learing_rate=0.001
    labels_input,images_input=get_data.get_data()
    print len(labels_input)
    print 'end read'




    images = tf.placeholder("float", shape=[None, 24,24,3])
    labels= tf.placeholder("float", shape=[None, 10])


    #卷积1

    w_conv1=weight_variable([5,5,3,64])
    b_conv1=bias_variable([64])
    h_conv1=tf.nn.relu(tf.nn.conv2d(images, w_conv1,[1,1,1,1],padding='SAME')+ b_conv1)# 24*24*64
    varable_list.append(w_conv1)
    varable_list.append(b_conv1)



    #池化1，正规化
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1') #12*12*64

    #卷积2
    w_conv2=weight_variable([5,5,64,64])
    b_conv2=bias_variable([64])
    h_conv2= (tf.nn.conv2d(h_norm1, w_conv2, [1, 1, 1, 1], padding='SAME')+b_conv2)


    #池化正规化2
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')#6*6*64
    varable_list.append(w_conv2)
    varable_list.append(b_conv2)


    #全连接层1

    w_fc1 = weight_variable([6 * 6 * 64, 384])
    b_fc1 = bias_variable([384])
    h_pool2_flat=tf.reshape(h_pool2,[-1,6*6*64])#

    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)#输出384
    varable_list.append(w_fc1)
    varable_list.append(b_fc1)
    #全连接2
    w_fc2=weight_variable([384,192])
    b_fc2=bias_variable([192])

    h_fc2=tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)#输出
    varable_list.append(w_fc2)
    varable_list.append(w_fc2)


    weight=weight_variable([192,10])
    bais=bias_variable([10])
    varable_list.append(weight)
    varable_list.append(bais)


    prediction=tf.matmul(h_fc2,weight)+bais
    #prediction是N*10的矩阵，为预测结果
    # labels是N*10的矩阵，是真实的结果


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=prediction))
    #cost=-tf.reduce_mean(labels*tf.log(prediction))
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(cost)
    saver=tf.train.Saver()
    

    sess = tf.Session()
    start_time=time.time()
    step_length=100
    start=0;
    end=start+step_length
    number_of_steps= len(labels_input) / step_length


    if (os.path.exists(check_point_path+".meta")):
        saver.restore(sess,check_point_path)       
    else:
        sess.run(tf.initialize_all_variables())#初始化变量,当从check point导入的时候不需要执行。


    for ranges in range(0,100):
        for w in range(0, number_of_steps):
            temp_images,temp_labels=getnext(start,end,images_input,labels_input)
            cost,train_step=sess.run([cost,train_step], feed_dict={images: temp_images, labels: temp_labels})
            t=time.time()-start_time
            x=t/(w + ranges * number_of_steps + 1)
            if (w+ranges*number_of_steps)%10==0:
                print ('step %d, loss = %.2f; %.3f sec/batch)' % (ranges * number_of_steps + w, cost, x))
            if(cost<4):
                learing_rate=0.0001
            if(cost<1):
                learing_rate=0.00001
            if (ranges*number_of_steps+w+1)%50==0:
                savepath=saver.save(sess, check_point_path)
                print "save at:"+savepath
            start+=step_length
            end+=step_length
            if(w==number_of_steps-1):
                start = 0;
                end = start + step_length
if __name__ == "__main__":
    start_train()



