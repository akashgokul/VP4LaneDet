import torch
import torch.nn as nn
torch.manual_seed(0)

#Testing VPGNet Arch without VP (for comparision purposes)

class LaneDetect(nn.Module):

    def __init__(self):
        super(LaneDetect, self).__init__()
        #Followed Figure 3 pg.5 of VPGNet paper
        self.shared = nn.Sequential(
            #Conv1
            nn.Conv2d(3, 96, kernel_size=8, stride=4, padding=0), #changed to 8 from 11
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(96),
            #Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(256),
            #Conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            #Conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            #Conv5
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #Conv6
            nn.Conv2d(384, 4096, kernel_size=6, stride=1, padding=3)
        )
        # self.grid_box = Sequential(
        #     #Conv 7
        #     nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0), 
        #     nn.Dropout(),
        #     #Conv 8
        #     nn.Conv2d(4096, 256, kernel_size=1, stride=1, padding=0), 
        #     #Tiling
        #     nn.ConvTranspose2d(256, 4, kernel_size = 8)
        # )
        self.obj_mask = nn.Sequential(
            #Conv 7
            nn.ConvTranspose2d(4096, 384, kernel_size=2, stride=2), 
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(384,256, kernel_size=2, stride=2), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(256,1 ,kernel_size=2, stride=2)
        )
        # self.vp = nn.Sequential(
        #     #Conv 7
        #     nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0), 
        #     nn.Dropout(),
        #     #Conv 8
        #     nn.Conv2d(4096, 320, kernel_size=1, stride=1, padding=0), 
        #     #Tiling
        #     #nn.ConvTranspose2d(320, 5, kernel_size = 8),
        # )
        
        
    def forward(self, x):


        #Forward pass through shared layers
        x = self.shared(x)

        #Pass through the obj_mask branch 
        obj_mask = torch.sigmoid(self.obj_mask(x))
        # #Reshape into (120,160,2)
        # obj_mask = obj_mask.view(-1,1,120,160)

        #Pass through the vp branch 
        # vp = torch.sigmoid(self.vp(x))
        # #Reshape into (120,160,5)
        # vp = vp.view(120,160,5)

        return obj_mask      #, vp