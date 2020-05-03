import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import scipy.io
import tqdm
import time
from lanedetect_model import LaneDetect

class LaneDetectionHelper:
    def __init__(self,
                model: VPGNet,
                loss = nn.BCELoss(),
                optimizer = None ,
                learning_rate = 1e-3):
        """
        Args:
            Model: LaneDetect instance to train/test
            Loss: Torch Loss function
            Optimizer: Torch optimizer for training
            Learning_rate: Learning rate for net (default = 1e-3)
        """
        #Checks that args are valid
        assert type(model) == LaneDetect

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device=self.device)   

        self.loss= loss

        if(optimizer == None):
            self.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        else:
            self.optimizer = optimizer

        self.learning_rate = learning_rate
    


    def train(self,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            num_epochs_vp: int):
        """
        Args:
            Train_dataloader: Dataloader for training dataset
            Val_dataloader: Dataloader for validation dataset
            Num_epochs_vp: Int (Number of Epochs to train)
        """

        #Checking that args are valid
        assert type(train_dataloader) == DataLoader
        assert type(validation_dataloader) == DataLoader
        assert num_epochs > 0

        self.model.train()
        for e in range(num_epochs):
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            self.optimizer.zero_grad()

            print("-----"*10)
            print("Training naive Lane Detection Model")
            print("-----"*10)
            num_batches = len(train_dataloader)
            for batch_number, (rgb_img, obj_mask ,_) in enumerate(train_dataloader):
                print("Training Batch: " + str(batch_number) + " / " + str(num_batches))
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.to(device=self.device)

                #Need for loss comp.
                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)

                obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = obj_mask_pred.to(device=self.device)
                loss = self.loss(obj_mask_pred.view(obj_mask.shape), obj_mask)


                loss.backward(retain_graph = True)
                self.optimizer.step()

                #Updating training accuracy and training loss
                train_loss += loss.item()
                #Using PIXEL-Wise Accuracy!
                obj_mask_pred = obj_mask_pred.view(vp.shape)
                train_acc += ((obj_mask_pred == obj_mask).sum().item() )  / (obj_mask.shape[0] * obj_mask.shape[1] * obj_mask.shape[2] * obj_mask.shape[3])

                self.optimizer.zero_grad()

            #Normalizing by number of batches
            train_loss = train_loss /  num_batches
            train_acc = 100 * train_vp_acc / num_batches

            val_obj_mask_loss, val_obj_mask_acc = self.eval(validation_dataloader)
            elapsed = time.time() - start_time
            print(
                "General Training: Epoch {:d} Train loss vp: {:.2f}. Train Accuracy VP: {:.2f}. Validation loss OBJ: {:.2f}. Validation Accuracy OBJ: {:.2f}. Elapsed time: {:.2f}ms. \n".format(
                e + 1, train_loss, train_acc, val_obj_mask_loss, val_obj_mask_acc, elapsed)
                )

        
    
    def eval(self, 
            dataloader: DataLoader):
        
        """
        Args:
            Dataloader: A torch dataloader
        
        Note: This function evaluates the model on the dataloader and returns obj_mask_loss, vp_loss, obj_mask_acc, vp_acc

        """
        self.model.eval()
        with torch.no_grad():
            obj_mask_loss = 0
            obj_mask_acc = 0.0

            num_batches = len(dataloader)
            for batch_number, (rgb_img,obj_mask, _) in enumerate(dataloader):
                print("Eval Batch: " + str(batch_number) + " / " + str(num_batches))
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.to(device=self.device)

                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)


                obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = obj_mask_pred.to(device=self.device)

                loss = self.loss(obj_mask_pred.view(obj_mask_pred.shape), obj_mask)

                obj_mask_loss += loss.item()

                obj_mask_pred = obj_mask_pred.view(obj_mask.shape)

                obj_mask_acc += ((obj_mask_pred == obj_mask).sum().item() )  / (obj_mask.shape[0] * obj_mask.shape[1] * obj_mask.shape[2]*obj_mask.shape[3])

                obj_mask_loss += loss_obj_mask

        obj_mask_loss = obj_mask_loss / num_batches

        obj_mask_acc = obj_mask_acc / num_batches

        return obj_mask_loss, 100 * obj_mask_acc


    #TODO: Update test function to get test set predictions (Currently very naive)
    def test(self, dataloader: DataLoader):
        """
        Args:
            Dataloader: A torch dataloader (assumes batchsz 1)
        
        Note: This function returns list of prediction of obj_maskmasks on testset

        """

        self.model.eval()
        with torch.no_grad():
            for batch_number, rgb_img in enumerate(dataloader):
                rgb_img = rgb_img.to(device = self.device)
                obj_mask_pred, obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = (obj_mask_pred > 0.5)
                obj_mask_pred = obj_mask_pred.cpu().numpy()
                rgb_img = rgb_img.cpu().numpy()
                temp_dict = {'img':rgb_img, 'obj_mask_pred': obj_mask_pred}
                scipy.io.savemat('test_pred/' + str(batch_number) + "_naive_pred.mat", temp_dict)

        print("Done Testing!")
        
        return


