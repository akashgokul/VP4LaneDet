import argparse
import torch
from torch.utils.data import DataLoader

from vp_lanedetect import VP4LaneDetection
from vpgnet_torch import VPGNet
from dataset import VPGData

from lanedetect_model import LaneDetect
from lanedetect import LaneDetectionHelper


def main(args):

    #TODO: Implement typical torch transform (e.g. normalize img)
    train_dataset = VPGData(args.root_dir, args.csv_path ,transform = None, split = 'train')
    valid_dataset = VPGData(args.root_dir, args.csv_path, transform = None, split = 'validation')

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 1)
    valid_dataloader = DataLoader(valid_dataset, batch_size = 1, shuffle = True, num_workers = 1)

    if(args.model == 'naive'):
        model = LaneDetect()
        helper = LaneDetectionHelper(model = model, learning_rate = args.learning_rate)
        helper.train(train_dataloader, valid_dataloader, args.num_epochs_general)
    
    else:
        model = VPGNet()
        helper = VP4LaneDetection(model = model, learning_rate = args.learning_rate)
        helper.train(train_dataloader, valid_dataloader, args.num_epochs_vp, args.num_epochs_general)
    
    test_dataset = VPGData(args.root_dir, args.csv_path, transform = None, split = 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 1)
    helper.test(test_dataloader)
    # test_loss, test_acc = helper.eval(test_dataloader)
    # print("Test Accuracy: " + str(test_acc))






if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vanishing Point for Lane Detection')

    parser.add_argument("--model", type=str, 
                        choices = ['naive','VPGNet'],default = 'VPGNet', 
                        help = 'Type of Model (naive = no vp, VPGNet = w/ VP)')

    #Data
    parser.add_argument('--root_dir', type=str,
                    help='Path of root dir containing data')
    
    parser.add_argument("--csv_path", type=str, help='Path of CSV file containing relative paths of imgs')


    # Model
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')

    parser.add_argument('--num_epochs_vp', type=int, default=5,
                        help='number of epochs for vp training phase')
    
    parser.add_argument('--num_epochs_general', type=int, default=5,
                        help='number of epochs for entire model (after vp phase)')

    parser.add_argument("--learning_rate", type=float, default = 1e-3, help='Learning Rate')


    args = parser.parse_args()
    print(args, end="\n\n")

    main(args)