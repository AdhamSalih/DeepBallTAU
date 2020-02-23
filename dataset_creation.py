import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

from torchvision import transforms, utils
import matplotlib.pyplot as plt

from torchvision import transforms, datasets

TRAIN_SET = ['1', '2', '3', '4']
TEST_SET = ['5', '6']
images_folder = 'E:/Deep Learning Project Files/Images1/'
annotations_folder = 'E:/Deep Learning Project Files/Annotations/'
images_sub_folder_format = "Film Role-0 ID-{set} T-2 m00s00-000-m00s00-185/"
csv_format = "cam-{set}.csv"

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, masks = sample['image'], sample['masks']
        #
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        # image = image.transpose((2, 0, 1))

        image = sample[0]
        masks = sample[1]
        image = image.transpose((2, 0, 1))
        return [torch.from_numpy(image.copy()),torch.from_numpy(masks.copy())]

class BallLandmarksDataset(Dataset):


    def __init__(self, csv_folder, csv_format, sets, root_dir, images_sub_folder_format, flip_option = False, transform=None, sequence_len = 1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        last_frames = np.zeros(len(sets))
        image_folders = [None] * len(sets)
        csv_files = [None] * len(sets)
        landmarks_frames = [None] * len(sets)
        last_frame_val = 0
        for i in range(len(sets)):
            set = sets[i]
            image_set_folder = root_dir + images_sub_folder_format.format(set = set)
            csv_file = csv_folder + csv_format.format(set = set)
            image_folders[i] = image_set_folder
            csv_files[i] = csv_file
            landmarks_frame = pd.read_csv(csv_file)
            last_frame_val += len(landmarks_frame)-5-(sequence_len-1)
            landmarks_frames[i] = landmarks_frame
            last_frames[i] = last_frame_val

        self.landmarks_frames = landmarks_frames
        self.image_folders = image_folders
        self.csv_files = csv_files
        self.last_frames = last_frames
        self.flip_option = flip_option
        self.transform = transform
        self.sequence_len = sequence_len
    def __len__(self):
        # return len(self.last_frames[-1])
        return (int(self.last_frames[-1]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        i = 0
        while idx >= self.last_frames[i] and i < len(self.last_frames):
            i += 1
        if i ==0:
            local_idx = idx
        else:
            local_idx = idx - self.last_frames[i-1]
        images = np.zeros([272, 480, 3*self.sequence_len]).astype(np.uint8)
        for j in range(self.sequence_len):
            frameNumber = self.landmarks_frames[i]['Frame No.'][local_idx+5 + j]
            x = self.landmarks_frames[i][' x'][local_idx+5+j]
            y = self.landmarks_frames[i][' y'][local_idx+5+j]
            image_name = str(frameNumber) + '.jpeg'
            # image = cv2.imread(self.image_folders[i] + image_name)
            image = cv2.imread(self.image_folders[i] + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] > 1080:
                print(image.shape)
                org_shape = image.shape
                crop = int((image.shape[0] - 1080)/2)
                image = image[crop:image.shape[0]-crop,:,:]
                print(crop, image.shape)
                if j == (self.sequence_len-1)/2:
                    masks = np.zeros(shape = (2,int(np.floor(image.shape[0]/16 + 0.5)), int(np.floor(image.shape[1]/16 + 0.5))))
                    print(x,y)
                    # if x != ' -' and int(y) > crop and int(y) <= org_shape[0] - crop:
                    if x != ' -' and int(y) > crop and int(y) <= org_shape[0] - crop and int(np.floor((int(x)-crop-1)/16 + 0.5)) < masks.shape[2]:# and (int(y)-crop-1)/16 < masks.shape[1]:
                        mask = np.zeros(shape = (int(np.floor(image.shape[0]/16 + 0.5)), int(np.floor(image.shape[1]/16 + 0.5))))
                        mask1 = np.ones_like(mask)
                        y_central = int(np.floor((int(y)-crop-1)/16 + 0.5))
                        x_central = int(np.floor((int(x)-crop-1)/16 + 0.5))
                        mask[y_central, x_central] = 1
                        mask1[y_central, x_central] = 0
                        kernel = np.ones((3,3),np.uint8)
                        if np.shape(mask) != 0:
                            mask1 = cv2.erode(mask1,kernel,iterations = 1)
                        masks[0,:,:] = mask
                        masks[1,:,:] = mask1
            else:
                if j == (self.sequence_len-1)/2:
                    masks = np.zeros(shape = (2,int(np.floor(image.shape[0]/16 +0.5)), int(np.floor(image.shape[1]/16 + 0.5))))
                    mask = np.zeros(shape = (int(np.floor(image.shape[0]/16 + 0.5)), int(np.floor(image.shape[1]/16 + 0.5))))
                    mask1 = np.ones_like(mask)
                    if x != ' -' and int(np.floor((int(x)-1)/16 + 0.5)) < masks.shape[2]:# and (int(y)-crop-1)/16 < masks.shape[1]:
                        y_central = int(np.floor((int(y)-1)/16 + 0.5))
                        x_central = int(np.floor((int(x)-1)/16 + 0.5))
                        mask[y_central, x_central] = 1
                        mask1[y_central, x_central] = 0
                        kernel = np.ones((3,3),np.uint8)
                        if np.shape(mask) != 0:
                            mask1 = cv2.erode(mask1,kernel,iterations = 1)
                    masks[0,:,:] = mask
                    masks[1,:,:] = mask1
            image = cv2.resize(image, dsize=(480, 272))
            if self.flip_option == True:
                image = np.fliplr(image)
            images[:, :, j*3:j*3+3] = image
            # sequnce_masks[j] = masks
        # sample = [image, masks]
        masks = masks.astype(np.uint8)
        if self.flip_option == True:
            masks[0] = np.fliplr(masks[0])
            masks[1] = np.fliplr(masks[1])
        sample = [images, masks]

        if self.transform:
            sample = self.transform(sample)

        return sample

def main():

    train_dataset = BallLandmarksDataset(csv_folder=annotations_folder, csv_format=csv_format, sets=TRAIN_SET, \
                                              root_dir=images_folder, images_sub_folder_format=images_sub_folder_format, flip_option= False, \
                                                sequence_len = 5, transform = ToTensor())

    test_dataset = BallLandmarksDataset(csv_folder=annotations_folder, csv_format=csv_format, sets=TEST_SET, \
                                              root_dir=images_folder, images_sub_folder_format=images_sub_folder_format, flip_option= False, \
                                                sequence_len = 5, transform = ToTensor())

    train_dataset_flip = BallLandmarksDataset(csv_folder=annotations_folder, csv_format=csv_format, sets=TRAIN_SET, \
                                              root_dir=images_folder, images_sub_folder_format=images_sub_folder_format, flip_option= True, \
                                                sequence_len = 5, transform = ToTensor())

    test_dataset_flip = BallLandmarksDataset(csv_folder=annotations_folder, csv_format=csv_format, sets=TEST_SET, \
                                              root_dir=images_folder, images_sub_folder_format=images_sub_folder_format, flip_option= True, \
                                                sequence_len = 5, transform = ToTensor())

    # sample = ball_dataset[0]
    # plt.figure()
    # plt.imshow(sample['image'])
    # # if (len(sample['landmarks']) != 0):
    # #     plt.scatter(sample['landmarks'][0][0], sample['landmarks'][0][1])
    # sample = ball_dataset[0]
    # plt.figure()
    # plt.imshow(sample['image'])
    # # if (len(sample['landmarks']) != 0):
    # #     plt.scatter(sample['landmarks'][0][0], sample['landmarks'][0][1])

    # sample = train_dataset[133]
    # plt.figure(figsize=(10, 6))
    # plt.imshow(sample['image'])
    # plt.figure(figsize=(10, 6))
    # plt.imshow(sample['masks'][0], cmap='gray')

    train_loader = DataLoader(train_dataset, batch_size=500)#, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=500)#, num_workers=4)

    train_loader_flip = DataLoader(train_dataset_flip, batch_size=500)#, num_workers=4)

    test_loader_flip = DataLoader(test_dataset_flip, batch_size=500)#, num_workers=4)
    # print(trainloader)
    # if (len(sample['landmarks']) != 0):
    # #     plt.scatter(sample['landmarks'][0][0], sample['landmarks'][0][1])
    # # # print(enumerate(trainloader))
    enumerate(train_loader)
    print(len(train_loader))
    trainSetList = [None] * len(train_loader)

    for i_batch, sample_batched in enumerate(train_loader):
        # if i_batch > 555:
        # print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())
        print(i_batch, sample_batched[0].size(),sample_batched[1].size())
        trainSetList[i_batch] = sample_batched
        batch_file = 'E:/Deep Learning Project Files/Batches_500_cropped_small_images_1_pixel_mask_sequence_5_stuck/train/train_batch_file' + str(i_batch) + '.h5'
        with open(batch_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(sample_batched, filehandle)

    for i_batch, sample_batched in enumerate(test_loader):
        # print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())
        print(i_batch, sample_batched[0].size(),sample_batched[1].size())
        trainSetList[i_batch] = sample_batched
        batch_file = 'E:/Deep Learning Project Files/Batches_500_cropped_small_images_1_pixel_mask_sequence_5_stuck/test/test_batch_file' + str(i_batch) + '.h5'
        with open(batch_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(sample_batched, filehandle)

    for i_batch, sample_batched in enumerate(train_loader_flip):
        # if i_batch > 555:
        # print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())
        print(i_batch, sample_batched[0].size(),sample_batched[1].size())
        trainSetList[i_batch] = sample_batched
        batch_file = 'E:/Deep Learning Project Files/Batches_500_cropped_small_images_1_pixel_mask_sequence_5_stuck/train/train_flip_batch_file' + str(i_batch) + '.h5'
        with open(batch_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(sample_batched, filehandle)

    for i_batch, sample_batched in enumerate(test_loader_flip):
        # print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())
        print(i_batch, sample_batched[0].size(),sample_batched[1].size())
        trainSetList[i_batch] = sample_batched
        batch_file = 'E:/Deep Learning Project Files/Batches_500_cropped_small_images_1_pixel_mask_sequence_5_stuck/test/test_flip_batch_file' + str(i_batch) + '.h5'
        with open(batch_file, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(sample_batched, filehandle)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())
    # for i_batch, sample_batched in enumerate(test_loader):
    #     print(i_batch, sample_batched['image'].size(),sample_batched['masks'].size())if __name__ == '__main__':
if __name__ == '__main__':
    main()
