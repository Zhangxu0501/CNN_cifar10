#coding:utf-8
import numpy as np
from scipy.misc import imsave
import struct
import cifar10

def process2(img):
    images=[]
    temp=[]
    for i in range(4,28):
        temp.append(img[i])

    return temp
def process(img):
    images=[]
    temp=[]
    for i in range(4,28):
        temp.append(img[i])
    for j in range(0,len(temp)):
        temp[j]=process2(temp[j])
    return temp



def get_data():
    file_num=5
    img_num=3000#没个文件读取的图片数
    pic_width=32
    pic_height=32
    files=[]
    labels=[]
    images=[]

    # for i in range(1,file_num+1):
    #     print("start read:" + str(i))
    #     f=open("/Users/zhangxu/Downloads/cifar-10-batches-bin/data_batch_" + str(i) + ".bin", 'rb')
    #     bytes=f.read()
    #     files.append(bytes)
    #     f.close()

    f=open(cifar10.data_path,'rb')
    files.append(f.read())
    f.close()

    k=1
    for file in files:
        k+=1
        for i in range(0,img_num):
            for j in range(0,1025):
                count=i * 3073 + j
                if((count%3073)==0):
                    labels.append(struct.unpack('B',file[count])[0])#以uint8格式解析
                else:
                    images.append([struct.unpack('B',file[i*3073+j])[0],struct.unpack('B',file[i*3073+j+1024])[0],struct.unpack('B',file[i*3073+j+2048])[0]])

    imgs = []
    for i in range(0,len(images)):
        images[i]=np.float32(images[i])

    for i in range(0,len(images)/1024):
        img = []
        for j in range(0,pic_width):
            img.append([])
            for k in range(0,pic_height):
                img[j].append(images[i*1024+j*pic_width+k])
        imgs.append(img)
    images=[]
    images=imgs
    labs=[]
    for i in range(0, len(labels)):
        labs.append([np.float64(0), np.float64(0), np.float64(0),np.float64(0), np.float64(0), np.float64(0), np.float64(0), np.float64(0), np.float64(0), np.float64(0)])
        labs[i][labels[i]] = np.float64(1)
    for i in range(0,len(images)):
        images[i]=process(images[i])
    print 'end process'
    return labs,images

def get_test_set():
    print "start read test set"
    img_num =1100# 没个文件读取的图片数
    pic_width = 32
    pic_height = 32
    files = []
    labels = []
    images = []


    f=open(cifar10.data_path,'rb')
    files.append(f.read())
    f.close()

    k = 1
    for file in files:
        k += 1
        for i in range(0, img_num):
            for j in range(0, 1025):
                count = i * 3073 + j
                if ((count % 3073) == 0):
                    labels.append(struct.unpack('B', file[count])[0])  # 以uint8格式解析
                else:
                    images.append(
                        [struct.unpack('B', file[i * 3073 + j])[0], struct.unpack('B', file[i * 3073 + j + 1024])[0],
                         struct.unpack('B', file[i * 3073 + j + 2048])[0]])

    imgs = []
    for i in range(0, len(images)):
        images[i] = np.float32(images[i])

    for i in range(0, len(images) / 1024):
        img = []
        for j in range(0, pic_width):
            img.append([])
            for k in range(0, pic_height):
                img[j].append(images[i * 1024 + j * pic_width + k])
        imgs.append(img)
    images=[]

    befor_process=imgs[:]

    for i in range(0,len(imgs)):
        imgs[i]=process(imgs[i])
    print 'end process'
    return labels, imgs,befor_process

def display(images):
    img=[]
    for i in range(len(images[0])):
        line = images[0][i]
        for j in range(1,len(images)):
            line.extend(images[j][i])
        img.append(line)
    imsave("test_set1.jpg",img)


    # a=Image.open("test_set.jpg")
    # a.show()
    #失效的包
def get_class(labels):
    cls=[]
    for i in labels:
        if i==0:
            cls.append("ariplane")
        elif i==1:
            cls.append("auto")
        elif i==2:
            cls.append("bird")
        elif i==3:
            cls.append("cat")
        elif i==4:
            cls.append("deer")
        elif i==5:
            cls.append("dog")
        elif i==6:
            cls.append("frog")
        elif i==7:
            cls.append("horse")
        elif i==8:
            cls.append("ship")
        elif i==9:
            cls.append("truck")
    return  cls