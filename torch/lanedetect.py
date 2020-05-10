import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import scipy.io
import time
from lanedetect_model import LaneDetect
from torchvision import transforms, utils

class LaneDetectionHelper:
    def __init__(self,
                model: LaneDetect,
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

        if(optimizer == None):
            self.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        else:
            self.optimizer = optimizer

        self.learning_rate = learning_rate
    


    def train(self,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            num_epochs: int):
        """
        Args:
            Train_dataloader: Dataloader for training dataset
            Val_dataloader: Dataloader for validation dataset
            Num_epochs: Int (Number of Epochs to train)
        """

        #Checking that args are valid
        assert type(train_dataloader) == DataLoader
        assert type(validation_dataloader) == DataLoader
        assert num_epochs > 0

        nimages = 0
        mean = 0.
        std = 0.

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

                #Normalizing img
                batch = rgb_img.view(rgb_img.size(0), rgb_img.size(1), -1).to(device=self.device)
                nimages += batch.size(0)
                mean += batch.mean(2).sum(0)
                std += batch.std(2).sum(0)

                # Final step
                mean /= nimages
                std /= nimages

                transform = transforms.Normalize(mean, std)
                rgb_img = transform(rgb_img).to(device=self.device)

                #Need for loss comp.
                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)
                print(obj_mask.shape)

                obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = obj_mask_pred.to(device=self.device)
                print(obj_mask_pred.shape)

                # loss_weights = 9 * obj_mask + torch.ones(obj_mask.shape).to(device=self.device)
                # loss_weights.to(device=self.device)
                # loss_func = torch.nn.BCELoss(weight = loss_weights)
                loss_func = torch.nn.BCEWithLogitsLoss()
                loss = loss_func(obj_mask_pred, obj_mask)
                print("------")


                loss.backward(retain_graph = True)
                self.optimizer.step()

                #Updating training accuracy and training loss
                train_loss += loss.item()
                #Using PIXEL-Wise Accuracy!
                round_obj_mask_pred = (obj_mask_pred > 0.5).float()
                train_acc += ((round_obj_mask_pred == obj_mask).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2])

                self.optimizer.zero_grad()

            #Normalizing by number of batches
            train_loss = train_loss /  num_batches
            train_acc = 100 * train_acc / num_batches

            val_obj_mask_loss, val_obj_mask_acc = self.eval(validation_dataloader)
            elapsed = time.time() - start_time
            print(
                "General Training: Epoch {:d} Train loss Obj: {:.2f}. Train Accuracy Obj: {:.2f}. Validation loss OBJ: {:.2f}. Validation Accuracy OBJ: {:.2f}. Elapsed time: {:.2f}ms. \n".format(
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
            nimages = 0
            mean = 0.
            std = 0.
            num_batches = len(dataloader)
            for batch_number, (rgb_img,obj_mask, _) in enumerate(dataloader):
                print("Eval Batch: " + str(batch_number) + " / " + str(num_batches))
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.type(torch.FloatTensor)

                #Normalizing img
                batch = rgb_img.view(rgb_img.size(0), rgb_img.size(1), -1).to(device=self.device)
                nimages += batch.size(0)
                mean += batch.mean(2).sum(0)
                std += batch.std(2).sum(0)

                # Final step
                mean /= nimages
                std /= nimages

                transform = transforms.Normalize(mean, std)
                rgb_img = transform(rgb_img).to(device=self.device)

                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)


                obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = obj_mask_pred.to(device=self.device)

                # loss_weights = 9 * obj_mask + torch.ones(obj_mask.shape).to(device=self.device)
                # loss_weights.to(device=self.device)
                # loss_func = torch.nn.BCELoss(weight = loss_weights)
                loss_func = torch.nn.BCEWithLogitsLoss()
                loss = loss_func(obj_mask_pred, obj_mask)

                round_obj_mask_pred = (obj_mask_pred > 0.5).float()
                obj_mask_acc += ((round_obj_mask_pred == obj_mask).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2])

                obj_mask_loss += loss.item()

        obj_mask_loss = obj_mask_loss / num_batches

        obj_mask_acc = obj_mask_acc / num_batches

        return obj_mask_loss, 100 * obj_mask_acc


    def test(self, dataloader: DataLoader):
        """
        Args:
            Dataloader: A torch dataloader (assumes batchsz 1)
        
        Note: This function returns list of prediction of obj_maskmasks on testset

        """
        nimages = 0
        mean = 0.
        std = 0.

        self.model.eval()
        with torch.no_grad():
            num_batches = len(dataloader)
            for batch_number, rgb_img in enumerate(dataloader):
                print("Training Batch: " + str(batch_number) + " / " + str(num_batches))
                rgb_img = rgb_img.type(torch.FloatTensor)

                #Normalizing img
                batch = rgb_img.view(rgb_img.size(0), rgb_img.size(1), -1).to(device=self.device)
                nimages += batch.size(0)
                mean += batch.mean(2).sum(0)
                std += batch.std(2).sum(0)

                # Final step
                mean /= nimages
                std /= nimages

                transform = transforms.Normalize(mean, std)
                rgb_img = transform(rgb_img).to(device=self.device)

                obj_mask_pred = self.model(rgb_img)
                obj_mask_pred = (obj_mask_pred > 0.5).float()
                obj_mask_pred = obj_mask_pred.cpu().numpy()

                rgb_img = rgb_img.cpu().numpy()

                temp_dict = {'img':rgb_img, 'obj_mask_pred': obj_mask_pred}
                scipy.io.savemat('naive_test_pred2/' + str(batch_number) + "_pred.mat", temp_dict)

        print("Done Testing!")
        
        return


