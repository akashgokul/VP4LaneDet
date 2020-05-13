#Torch dataset to perform pre-processing of VPGNet data
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import math
import numpy as np 
import pandas as pd
import cv2
from scipy import misc, io

class VPGData(Dataset):
    def __init__(self, rootdir: str, csv_path: str, transform=None, split = 'train'):
        """
        Args: 
            Rootdir: String of rootdir to the  (Assumes DOES NOT end with '/')
            Csv_path: String of directory of csv file containing paths of all images
                - This CSV file should contain the RELATIVE PATH of the imgs (don't worry about root)

            Tranform: Torch.transform to be performed on data 

            Split: String denoting the type of dataset (train, validation , test). Currently 80-20 split
        """
        
        assert rootdir[-1] != '/'
        split = split.lower()
        assert split in ['train', 'validation', 'test']

        self.rootdir = rootdir

        #Reads csv
        self.df_from_csv = pd.read_csv(csv_path)

        
        #On Savio some data missing, so using this
        row2delete = []
        for index, row in self.df_from_csv.iterrows():
            img_name = row[0]
            if(not os.path.exists(self.rootdir + img_name)):
                row2delete.append(index)
        self.df_from_csv = self.df_from_csv.drop(index=row2delete)
        self.num_imgs = len(self.df_from_csv.index)

        #Shuffles rows
        np.random.seed(0)
        self.df_from_csv = self.df_from_csv.iloc[np.random.permutation(self.num_imgs)]

        #setting split
        self.split = split

        #Assigning test set
        self.num_test_samples = math.floor(0.2 * self.num_imgs)
        self.test_img_names = self.df_from_csv.iloc[self.num_test_samples:]

        train_and_valid_img_names = self.df_from_csv.iloc[:self.num_test_samples]

        #Assigning train set
        self.num_train_samples = math.floor(0.6 * len(train_and_valid_img_names.index))
        self.train_img_names = train_and_valid_img_names.iloc[0 : self.num_train_samples]

        #Assigning validation set
        self.validation_img_names = train_and_valid_img_names.iloc[self.num_train_samples:]
        self.num_validation_samples = len(self.validation_img_names.index)

        #This is just the normal torch transform (e.g. Normalizing image, etc. )
        #NOT the data preprocessing (e.g. shift lane labels) we need to do!
        self.transform = transform
    
    def __len__(self):
        if(self.split == 'train'):
            return self.num_train_samples
        elif(self.split == 'validation'):
            return self.num_validation_samples
        else:
            return self.num_test_samples
    
    def __getitem__(self,idx):

        #Gets the img corresponding to idx and split
        if(self.split == 'train'):
            img_name = self.train_img_names.iloc[idx][0]
        elif(self.split == 'validation'):
            img_name = self.validation_img_names.iloc[idx][0]
        else:
            img_name = self.test_img_names.iloc[idx][0]
        
        #Read image
        img_path = self.rootdir + img_name
        #This creates a dictionary of mat file contents
        mat_dict = io.loadmat(img_path)
        
        #Get 5 channel img
        img = mat_dict['rgb_seg_vp']

        #Slices first 3 channels for the rgb portion of img
        rgb_img = img[:,:,:3]
        # rgb_img = np.rollaxis(rgb_img, 2, 0) 
        # rgb_img = rgb_img.astype(np.float32)

        #Slices only the 4th channel (0-indexed) for the obj mask
        obj_mask = img[:, :, 3]
        
        #Creates binary mask mapping {0,7} -> {0,1}
        f = lambda x: 1 if (x >= 1 and x <=7) else 0
        f_func = np.vectorize(f)
        obj_mask = f_func(obj_mask)
        #Resize into 120x160 the size of output
        #obj_mask = np.resize(obj_mask, (120,160))
        obj_mask = obj_mask.astype(np.float32)
        
        #grid level annotation
        grid = np.zeros_like(obj_mask)
        for i in range(480):
          for j in range(640):
            if obj_mask[i][j] == 1:
              grid[i][j] = 1
              for k in range(3):
                for l in range(3):
                  if i-k-1 >= 0  and j-l-1 >= 0:
                    grid[i-k-1][j-l-1] = 1
              for k in range(3):
                for l in range(3):
                  if i-k-1 >= 0  and j+l+1 < 640:
                    grid[i-k-1][j+l+1] = 1
              for k in range(3):
                for l in range(3):
                  if i+k+1 < 480  and j-l-1 >= 0:
                    grid[i+k+1][j-l-1] = 1
              for k in range(3):
                for l in range(3):
                  if i+k+1 < 480  and j+l+1 < 640:
                    grid[i+k+1][j+l+1] = 1
        obj_mask = grid
        # obj_mask_copy = obj_mask #temporary copy
        # obj_mask_copy = obj_mask_copy.astype(int)
        obj_mask = cv2.resize(obj_mask, dsize = (160,120),interpolation=cv2.INTER_CUBIC)
        obj_mask = obj_mask.astype(int)
        

        #Commented out below b/c only one_channel obj_mask
        #Repeating for the second channel which inverts all values in first channel computed above ^
        # g = lambda x: 1 if (x == 0) else 0
        # g_func = np.vectorize(g)
        # obj_channel_2 = np.copy(obj_mask)
        # obj_mask_channel_2 = g_func(obj_channel_2)
        # #obj_mask_channel_2 = np.resize(obj_mask_channel_2, (120,160))
        # obj_mask_channel_2 = obj_mask_channel_2.astype(np.float32)
        # obj_mask_channel_2 = cv2.resize(obj_mask_channel_2, dsize = (160,120),interpolation=cv2.INTER_CUBIC)
        # obj_mask_channel_2 = obj_mask_channel_2.astype(int)
        
        #Stacks the channels together to create (120,60,2)
        # obj_mask = np.dstack((obj_mask, obj_mask_channel_2))
        # obj_mask = np.rollaxis(obj_mask, 2, 0) 


        #Extracts 5th dimension to get vpp
        vp  = img[:,:,4]

        #Maps VP to only count easy (Map is {0,2} -> {0,1})
        h = lambda x: 1 if (x == 1) else 0
        h_func = np.vectorize(h)
        vp = h_func(vp)
        #vp = np.resize(vp, (120,160))
        zero_vp = np.zeros_like(vp)

        #If no vp exists case:
        #Creates 4 channels of zero and final channel of all 1
        if(np.array_equal(vp, zero_vp)):
            vp = np.resize(vp, (120,160))
            zero_vp = np.zeros_like(vp)
            vp = np.dstack((zero_vp, zero_vp, zero_vp, zero_vp, np.ones_like(zero_vp)))

        #Case where VP exists
        else:
            #Fifth channel is all zero b/c if VP exists => absence channel = 0
            

            #Finds pixel coordinates of vp
            row_vp, col_vp = np.nonzero(vp)
            row_vp = row_vp[0]
            col_vp = col_vp [0]
            row_vp = int(row_vp/4)
            col_vp = int(col_vp/4)
            vp = np.resize(vp, (120,160))
            zero_vp = np.zeros_like(vp)
            first_channel_upperL_corner = np.zeros_like(vp)
            second_channel_upperR_corner = np.zeros_like(vp)
            third_channel_lowerL_corner = np.zeros_like(vp)
            forth_channel_lowerR_corner = np.zeros_like(vp)
            fifth_channel = zero_vp
            
            for i in range(120):
              for j in range(160):
           
                if i <= row_vp and j <= col_vp:
                  #Creates first channel corresp to upper left corner relative to VP
                  first_channel_upperL_corner[i][j] = 1
                if i <= row_vp and j >= col_vp:
                  #Creates second channel corresp to upper right corner relative to VP
                  second_channel_upperR_corner[i][j] = 1
                if i >= row_vp and j <= col_vp:
                  #Creates third channel corresp to lower left corner relative to VP
                  third_channel_lowerL_corner[i][j] = 1
                if i >= row_vp and j >= col_vp:
                  #Creates fourth channel corresp to lower right corner relative to VP
                  forth_channel_lowerR_corner[i][j] = 1

            #first_channel_upperL_corner = np.ones((row_vp, col_vp, 1))
            #second_channel_upperR_corner = np.ones((row_vp,vp.shape[1]-col_vp, 1))
            #third_channel_lowerL_corner = np.ones((vp.shape[0] - row_vp, col_vp, 1))
            #forth_channel_lowerR_corner = np.ones((vp.shape[0] - row_vp, vp.shape[1]-col_vp, 1)
            #Stacks all the channels above
            vp = np.dstack((first_channel_upperL_corner, second_channel_upperR_corner, third_channel_lowerL_corner, forth_channel_lowerR_corner, fifth_channel))

        # vp = np.rollaxis(vp, 2, 0) 

        # obj_mask = obj_mask.astype(np.float32)
        # vp = vp.astype(np.float32) 

        if(self.transform):
            rgb_img = self.transform(rgb_img)

        # if(self.split == 'test'):
        #     return rgb_img
        
        tensor_transform = transforms.ToTensor()
        return rgb_img, tensor_transform(obj_mask), tensor_transform(vp)



