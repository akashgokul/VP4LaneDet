import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import scipy.io
import time
from vpgnet_torch import VPGNet

class VP4LaneDetection:
    def __init__(self,
                model: VPGNet,
                loss_vp = nn.BCELoss(),
                loss_obj_mask = nn.BCELoss(),
                optimizer = None ,
                learning_rate = 1e-4):
        """
        Args:
            Model: VPGNet instance to train/test
            Loss_vp: Torch loss function for training the vp branch
            loss_obj_mask: Torch loss function for the overall net (after vp training phase)
            Optimizer: Torch optimizer for training
            Learning_rate: Learning rate for net (default = 1e-3)
        """
        #Checks that args are valid
        assert type(model) == VPGNet

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device=self.device)

        self.loss_vp = loss_vp
        self.loss_obj_mask = loss_obj_mask

        if(optimizer == None):
            self.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        else:
            self.optimizer = optimizer

        self.learning_rate = learning_rate
    


    def train(self,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            num_epochs_vp: int,
            num_epochs_general: int):
        """
        Args:
            Train_dataloader: Dataloader for training dataset
            Val_dataloader: Dataloader for validation dataset
            Num_epochs_vp: Number of epochs to train vpp branch (cross-entropy)
            Num_epochs_general: Number of epochs to train entire model (L1 obj_mask Loss)
        """

        """
        Args:
            Train_dataloader: Dataloader for training dataset
            Val_dataloader: Dataloader for validation dataset
            Num_epochs_vp: Number of epochs to train vpp branch (cross-entropy)
            Num_epochs_general: Number of epochs to train entire model (L1 obj_mask Loss)
        """

        #Checking that args are valid
        assert type(train_dataloader) == DataLoader
        assert type(validation_dataloader) == DataLoader
        assert num_epochs_vp >= 0
        assert num_epochs_general >= 0

        vp_phase_train_loss = []
        vp_phase_train_acc = []
        vp_phase_val_loss = []
        vp_phase_val_acc = []

        self.model.train()
        for e in range(num_epochs_vp):
            start_time = time.time()
            train_loss = 0.0
            train_vp_acc = 0.0
            self.optimizer.zero_grad()

            print("-----"*10)
            print("VP Training Phase")
            print("-----"*10)
            num_batches = len(train_dataloader)
            for batch_number, (rgb_img, obj_mask ,vp) in enumerate(train_dataloader):
                if batch_number%5 == 0:
                  print("Training Batch: " + str(batch_number) + " / " + str(num_batches))

                
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.to(device=self.device)

                #Need for loss comp.
                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)


                #Need for loss comp.
                vp = vp.type(torch.FloatTensor)
                # vp_2 = (vp[1]).numpy()
                vp = vp.to(device=self.device)
                
                 #weight calculation
                weights = 3*vp + 1
                weights = weights.to(device=self.device)

                #print("unique tensor: ",torch.unique(vp))
                outputs = self.model(rgb_img)
                obj_mask_pred = outputs[0]
                
                vp_pred = outputs[1]
                
                #vptiling
  
                # b = (vp_pred.size())[0]
                # tiling_vp = outputs[2]
                # #Haozhi Version
                # for z in range (b):
                #   for i in range(15):
                #     for j in range(20):
                #       vp_pred[z,0,i*8:i*8+8,j*8:j*8+8] = torch.flatten(tiling_vp[z,0:64,i,j]).view(8,8)
                #       vp_pred[z,1,i*8:i*8+8,j*8:j*8+8] = torch.flatten(tiling_vp[z,64:128,i,j]).view(8,8)
                #       vp_pred[z,2,i*8:i*8+8,j*8:j*8+8] = torch.flatten(tiling_vp[z,128:192,i,j]).view(8,8)
                #       vp_pred[z,3,i*8:i*8+8,j*8:j*8+8] = torch.flatten(tiling_vp[z,192:256,i,j]).view(8,8)


                
                #My Version
                # for z in range(b):
                #   count = 0
                #   for i in range(4):
                #     for j in range(8):
                #       for k in range(8):
                #         a = vp_pred[z,count,:,:]
                #         tiling_vp[z,i,j*15:j*15+15,k*20:k*20+20] = a
                #         count = count+1

                # vp_pred = tiling_vp


                # vp_pred_2 = torch.tensor((vp_pred[1])).cpu().numpy()
                # temp_dict = {'vp':(vp_2),'vp_pred':(vp_pred_2)}
                # scipy.io.savemat('/content/gdrive/My Drive/VPGNet/test/' + str(batch_number) + ".mat", temp_dict)
                #print(vp_pred.size())

                obj_mask_pred = obj_mask_pred.to(device=self.device)

                vp_pred = vp_pred.to(device=self.device)

                
                #weight calculation
                #print(row_vp)
               
                      
                #print("vp",vp.size())
                #print("vp_pred",vp_pred.size())
                self.loss_vp = nn.BCELoss(weight=weights)
                loss_vp = self.loss_vp(vp_pred, vp)
                #print(loss_vp.item())

                #CHECK THIS BELOW!!!!
                loss_vp.backward(retain_graph = True)
                self.optimizer.step()

                #Updating training accuracy and training loss
                train_loss += loss_vp.item()
               
                #round_pred_vp = torch.sigmoid(vp_pred)
                #print(torch.unique(round_pred_vp))
                #print(round_pred_vp.shape == vp_pred.shape)
                if batch_number%5 == 0:
                  print(loss_vp.item())
                  print(1-((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3]))
                train_vp_acc += (1-(((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3])))
                #print(train_vp_acc)

                self.optimizer.zero_grad()

            #Normalizing by number of batches
            train_loss = train_loss /  num_batches
            train_vp_acc = 100 * train_vp_acc / num_batches

            val_obj_mask_loss, validation_loss_vp, val_obj_mask_acc, validation_acc_vp = self.eval(validation_dataloader)
            
            vp_phase_train_loss.append(train_loss)
            vp_phase_train_acc.append(train_vp_acc)
            vp_phase_val_acc.append(validation_acc_vp)
            vp_phase_val_loss.append(validation_loss_vp)
            
            
            elapsed = time.time() - start_time
            print(
                "General Training: Epoch {:d} Train loss VP: {:.2f}. Train Accuracy VP: {:.2f}. Validation loss OBJ: {:.2f}. Validation Accuracy OBJ: {:.2f}. Validation Loss VP: {:.2f}. Validation Accuracy VP: {:.2f}. Elapsed time: {:.2f}ms. \n".format(
                e + 1, train_loss, train_vp_acc, val_obj_mask_loss, val_obj_mask_acc, validation_loss_vp, validation_acc_vp, elapsed)
                )
            # print(
            #      "General Training: Epoch {:d} Train loss VP: {:.2f}. Train Accuracy VP: {:.2f} . Elapsed time: {:.2f}ms. \n".format(
            #      e + 1, train_loss, train_vp_acc, elapsed)
            #     )

        phase2_vp_train_acc = []
        phase2_vp_val_acc = []

        phase2_mask_train_acc = []
        phase2_mask_val_acc = []

        phase2_loss = []
        torch.cuda.empty_cache()
        for e in range(num_epochs_general):
            start_time = time.time()
            train_loss = 0
            train_acc_vp_p2 = 0.0
            train_acc_obj = 0.0
            w1 = 1
            w4 = w1
            num_batches = len(train_dataloader)
            self.optimizer.zero_grad()
            print("----"*10)
            print("Complete Net Training Phase (Phase II)")
            print("----"*10)
            for batch_number, (rgb_img, obj_mask, vp) in enumerate(train_dataloader):
                if batch_number%5 == 0:
                  print("Training Batch: " + str(batch_number) + " / " + str(num_batches))

                torch.autograd.set_detect_anomaly(True)
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.to(device=self.device)

                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)

                vp = vp.type(torch.FloatTensor)
                vp = vp.to(device=self.device)

                outputs = self.model(rgb_img)

                obj_mask_pred = outputs[0]
                obj_mask_pred = obj_mask_pred.to(device=self.device)

                vp_pred = outputs[1]
                vp_pred = vp_pred.to(device=self.device)

                weights = 3*vp + 1
                weights = weights.to(device=self.device)
                self.loss_vp = nn.BCELoss(weight = weights)

                weights2 = 75*obj_mask + 1
                weights2 = weights2.to(device=self.device)
                self.loss_obj_mask = nn.BCELoss(weight = weights2)

                loss_vp = self.loss_vp(vp_pred, vp)
                loss_obj_mask = self.loss_obj_mask(obj_mask_pred,obj_mask)
                if(batch_number == 0):
                    w1 = 1 / loss_obj_mask.item()
                    w4 = 1 / loss_vp.item()

                loss = w1*loss_obj_mask + w4*loss_vp
                loss.backward(retain_graph = True)
                self.optimizer.step()
                train_loss+= loss.item()

                if(batch_number%5==0):
                  print("VP Loss: ",loss_vp.item())
                  print(1-((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3]))

                  print("Mask Loss:",loss_obj_mask.item())
                  print(1-((abs(obj_mask_pred-obj_mask)).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*vp_pred.shape[3]))
                train_acc_vp_p2 += (1-((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3]))
                train_acc_obj += (1-((abs(obj_mask_pred-obj_mask)).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*vp_pred.shape[3]))
                self.optimizer.zero_grad()

            #Normalizing by number of batches
            train_loss = train_loss /  num_batches
            train_acc_vp_p2 = 100 * train_acc_vp_p2 / num_batches
            train_acc_obj = 100 * train_acc_obj / num_batches


            phase2_loss.append(train_loss)
            phase2_vp_train_acc.append(train_acc_vp_p2)
            phase2_mask_train_acc.append(train_acc_obj)

            val_obj_mask_loss, validation_loss_vp, val_obj_mask_acc, validation_acc_vp = self.eval(validation_dataloader)
            
            phase2_vp_val_acc.append(validation_acc_vp)
            phase2_mask_val_acc.append(val_obj_mask_acc)
            
            
            elapsed = time.time() - start_time
            print(
                "General Training: Epoch {:d} Train loss: {:.2f}. Train Accuracy Obj Mask: {:.2f}. Train Accuracy VP: {:.2f}. Validation loss OBJ: {:.2f}. Validation Accuracy OBJ: {:.2f}. Validation Loss VP: {:.2f}. Validation Accuracy VP: {:.2f}. Elapsed time: {:.2f}ms. \n".format(
                e + 1, train_loss, train_acc_obj, train_acc_vp_p2, val_obj_mask_loss, val_obj_mask_acc, validation_loss_vp, validation_acc_vp, elapsed)
                )
            
        
        np.save("phase1loss.npy",np.array(vp_phase_train_loss))
        np.save("phase1_vp_train_acc.npy",np.array(vp_phase_train_acc))
        np.save("phase1_vp_val_loss.npy",np.array(vp_phase_val_loss))
        np.save("phase1_vp_val_acc.npy",np.array(vp_phase_val_acc))

        np.save("phase2loss.npy",np.array(phase2_loss))
        np.save("phase2_vp_train_acc.npy",np.array(phase2_vp_train_acc))
        np.save("phase2_vp_val_acc.npy",np.array(phase2_vp_val_acc))

        np.save("phase2_mask_train_acc.npy",np.array(phase2_mask_train_acc))
        np.save("phase2_mask_val_acc.npy",np.array(phase2_mask_val_acc))
    
    
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
            vp_loss = 0
            vp_acc = 0.0
            obj_mask_acc = 0.0

            num_batches = len(dataloader)
            for batch_number, (rgb_img,obj_mask, vp) in enumerate(dataloader):
                if(batch_number%200==0):
                  print("Eval Batch: " + str(batch_number) + " / " + str(num_batches))
                rgb_img = rgb_img.type(torch.FloatTensor)
                rgb_img = rgb_img.to(device=self.device)

                obj_mask = obj_mask.type(torch.FloatTensor)
                obj_mask = obj_mask.to(device=self.device)

                vp = vp.type(torch.FloatTensor)
                vp = vp.to(device=self.device)

                weights = 3*vp + 1
                weights = weights.to(device = self.device)

                weights2 = 75*obj_mask + 1
                weights2 = weights2.to(device = self.device)

                outputs = self.model(rgb_img)

                obj_mask_pred = outputs[0]
                obj_mask_pred = obj_mask_pred.to(device=self.device)

                vp_pred = outputs[1]
                vp_pred = vp_pred.to(device=self.device)
                
                
                
                self.loss_vp = nn.BCELoss(weight = weights)
                self.loss_obj_mask = nn.BCELoss(weight = weights2)
                loss_vp = self.loss_vp(vp_pred, vp)
                loss_obj_mask = self.loss_obj_mask(obj_mask_pred,obj_mask)

                vp_loss += loss_vp.item()
                obj_mask_loss += loss_obj_mask.item()


                #round_pred_obj = (obj_mask_pred > 0.5).float()
                #round_pred_vp = (vp_pred > 0.5).float()
                #round_pred_vp = torch.sigmoid(vp_pred)

                vp_acc += (1-((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3]))
                obj_mask_acc += (1-((abs(obj_mask_pred-obj_mask)).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*obj_mask_pred.shape[3]))

                obj_mask_loss += loss_obj_mask
                vp_loss += loss_vp

        obj_mask_loss = obj_mask_loss / num_batches
        vp_loss = vp_loss / num_batches

        vp_acc = vp_acc / num_batches
        obj_mask_acc = obj_mask_acc / num_batches

        return obj_mask_loss, vp_loss , 100 * obj_mask_acc, 100*vp_acc



    def test(self, dataloader: DataLoader):
        """
        Args:
            Dataloader: A torch dataloader (assumes batchsz 1)
        
        Note: This function returns list of prediction of obj_maskmasks on testset

        """

        self.model.eval()
        with torch.no_grad():
            for batch_number, (rgb_img,obj_mask,vp) in enumerate(dataloader):
                if(batch_number%1==0):
                  print("Test Batch: " + str(batch_number))
                rgb_img = rgb_img.to(device = self.device)
                obj_mask_pred, vp_pred = self.model(rgb_img)
                #obj_mask_pred = (obj_mask_pred > 0.5)
                obj_mask_pred = obj_mask_pred.cpu().numpy()
                #print(torch.unique(vp_pred))
                #vp_pred = (vp_pred > 0.5)
                #vp_pred = torch.sigmoid(vp_pred)
                vp = vp.cpu().numpy()
                obj_mask = obj_mask.cpu().numpy()
                vp_pred = vp_pred.cpu().numpy()
                rgb_img = rgb_img.cpu().numpy()
                rgb_img = np.rollaxis(rgb_img, 0, 2) 
                #print('np',np.unique(vp_pred))
                print("VP Loss: ")
                print(1-((abs(vp_pred - vp)).sum().item() )  / (vp_pred.shape[0] * vp_pred.shape[1] * vp_pred.shape[2] * vp_pred.shape[3]))

                print("Mask Loss:")
                print(1-((abs(obj_mask_pred-obj_mask)).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*vp_pred.shape[3]))
                temp_dict = {'img':rgb_img,'obj_mask':obj_mask, 'obj_mask_pred': obj_mask_pred,'vp':vp, 'vp_pred':vp_pred}
                #scipy.io.savemat('/content/gdrive/My Drive/VPGNet/test_pred/' + str(batch_number) + "_pred.mat", temp_dict)

        print("Done Testing!")
        
        return