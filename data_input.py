import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import fastaniso

class database:
    '''path includes all required paths in data and annotation extraction'''
    path = {'train_data': './NIO_training_resized', 'train_annotation': './labels_training_resized1',
            'test_data': './NIO_validation_resized', 'test_annotation': './labels_validation_resized'}

    batch_offset = 0
    epochs_completed = 0
    def __init__(self, img_option={},epoch = 25):
        '''
        :param img_option:
        img_option = {'resize_tag': True/ False, 'resize_size'=int, 'weight': int,'operation':string(can be sobel, laplacian)}
        'resize_size' is the size of resized output img
        'weight' is the penalized weight to solve the unbalanced dataset problem(defect pixels are much fewer than defect)

        '''
        self.img_option = img_option
        self.epoch = epoch

    def get_path(self, path_):
        return [os.path.join(path_,f) for f in os.listdir(path_) if f.endswith('png') or f.endswith('tif')]

    def data_extraction(self,data_path):

        '''
        :param data_path: the path for the to be extracted data
        :return: image in the range of (0,255)
        some gradient-based pre-processing methods can be done to enhance the quality of the images
        normalization is performed to each image to leave the gv values in the same range
        '''
        data_path_list = self.get_path(data_path)
        data = []
        for p in data_path_list:
            img = cv2.imread(p,-1)
            if self.img_option['resize_tag']:
                size = self.img_option['resize_size']
                img = cv2.resize(img,(size,size))
            if 'operation' in self.img_option:
                img = self.image_operation(img)
            img = np.expand_dims(img,axis=-1)
            img = np.concatenate((img,img,img),axis=-1)
            # data normalization
            data.append((img-img.min())/(img.max()-img.min()))
        data = np.array(data,dtype=np.float32)
        return 255*data

    def annotation_extraction(self,label_path):
        '''
        :param label_path: path of the annotation
        :return: annotation map
        the last dimension of the return is 2, because I want to re-weight the background pixels and the object pixels.
        data_labels[i,j,k,0] = 1 means that this pixel belongs to background
        data_labels[i,j,k,1] = 1 means that this pixel belongs to object
        '''
        label_path_list = self.get_path(label_path)
        data_labels = []
        weight = int(self.img_option['weight'])
        for p in label_path_list:
            img_label = cv2.imread(p,-1)
            if len(np.shape(img_label)) > 2:
                img_label = img_label[:,:,0]
            #print('.......................')
            #print(np.shape(img_label))
            if self.img_option['resize_tag']:
                size = self.img_option['resize_size']
                img_label = cv2.resize(img_label,(size,size))
            img_label = img_label/img_label.max()
            temp_img = img_label.copy()
            temp_img = np.expand_dims(temp_img,-1)
            temp_img = np.concatenate((temp_img,temp_img,temp_img),axis=-1)
            temp_img[:,:,0] = -(temp_img[:,:,0]-1)
            temp_img[:,:,1] = temp_img[:,:,1]*weight
            # img_label = np.squeeze(img_label)
            data_labels.append(temp_img[:,:,0:2])
        data_labels = np.array(data_labels,dtype=np.float32)
        return data_labels

    # data extraction for segmentation
    def data_segmentation(self):
        # training data extraction
        train_data_path = self.path['train_data']
        self.train_data = self.data_extraction(train_data_path)

        # testing data extraction
        train_data_path= self.path['test_data']
        self.test_data = self.data_extraction(train_data_path)

        # extract training labels for per-pixel (annotation for triandata)
        # for last the dimension, index 0 for background and 1 for defect region
        train_annotation_path = self.path['train_annotation']
        self.train_annotation = self.annotation_extraction(train_annotation_path)

        # extract test labels for per-pixel (annotation for testdata)
        test_annotation_path = self.path['test_annotation']
        self.test_annotation = self.annotation_extraction(test_annotation_path)

        return self.train_data, self.test_data, self.train_annotation, self.test_annotation

    def fetch_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.train_data.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.train_data.shape[0])
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_annotation = self.train_annotation[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.train_data[start:end], self.train_annotation[start:end]

    def image_operation(self,image):
        '''

        :param image: data image
        :return: image after some pre-processing
        '''
        if self.img_option['operation'] == 'sobelx':
            processed_img = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        elif self.img_option['operation'] == 'sobely':
            processed_img = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        elif self.img_option['operation'] == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            processed_img = np.sqrt(sobelx*sobelx + sobely*sobely)
        elif self.img_option['operation'] == 'laplacian':
            processed_img = cv2.Laplacian(image, cv2.CV_32F)
        else:
            raise Exception('this operation is not included')

        return processed_img










