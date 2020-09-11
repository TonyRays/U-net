from matplotlib import pyplot as plt
from monai import transforms as mn_tf
import cv2
from PIL import Image

def read_single_dict():
    # image = '../../dataset/LCTSC_1/test_nii_1/LCTSC-Test-S1-101.nii.gz'
    # label = '../../dataset/LCTSC_1/test_mask_nii/LCTSC-Test-S1-101.nii.gz'
    # set image path
    image = 'D:\\code\\U-net\\data\\imgs\\COVID-19-CT-Seg_20cases\\coronacases_org_001.nii'
    # set label path
    label = 'D:\\code\\U-net\\data\\masks\\Lung_and_Infection_Mask\\coronacases_001.nii'
    keys = ('image', 'label')
    mn_tfs = mn_tf.Compose([
        mn_tf.LoadNiftiD(keys),
        # mn_tf.AsChannelFirstD('image'),
        mn_tf.AddChannelD(keys),
        # mn_tf.SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
        mn_tf.OrientationD(keys, axcodes='RAS'),
        mn_tf.ThresholdIntensityD('image', threshold=600, above=False, cval=600),
        mn_tf.ThresholdIntensityD('image', threshold=-1000, above=True, cval=-1000),

        #等比例压缩，压缩到0到255
        mn_tf.ScaleIntensityD('image', minv=0.0, maxv=255.0), # show image

        mn_tf.ScaleIntensityD('image'),

        #改到512*512
        mn_tf.ResizeD(keys, (512, 512, -1), mode=('trilinear', 'nearest')),
        mn_tf.AsChannelFirstD(keys),
        # mn_tf.RandAffineD(keys, spatial_size=(-1, -1, -1),
        #                   rotate_range=(0, 0, np.pi / 2),
        #                   scale_range=(0.1, 0.1),
        #                   mode=('bilinear', 'nearest'),
        #                   prob=1.0),
        mn_tf.ToTensorD(keys)

    ])
    data_dict = mn_tfs({'image': image, 'label': label})
    print(data_dict['image'].shape, data_dict['label'].shape)
    slices = data_dict['image']
    masks = data_dict['label']

    # x = [1, 2, 3, 4, 5]
    #
    # y = [10, 5, 15, 10, 20]
    #
    # plt.plot(x, y, 'ro-', color='blue')
    #
    # plt.savefig('testblueline.jpg')

    plt.show()

    for idx, item in enumerate(zip(slices, masks)):
        image = item[0][0]
        label = item[1][0] * 255
        #index = torch.where(label == 1275)
        # plt.savefig('D:\code\Pytorch-UNet-master\data_test\mask'+str(idx)+'.jpg')
        # path_test =  'D:\\code\\U-net\\train_images_1\\1\\img\\' + str(idx) + '_image' + '.png'
        # print(path_test)
        plt.imsave('D:\\code\\U-net\\train_images_1\\0\\img\\' + str(idx)  + '.png', image, cmap='gray')
        plt.imsave('D:\\code\\U-net\\train_images_1\\0\\mask\\' + str(idx) + '_mask' + '.png', label, cmap='gray')
        if idx >= 120 and idx <= 121:
            # print(image)
            print(image.min(), image.max())
            plt.imshow(image, cmap='gray')
            # plt.imshow(image)
            plt.show()
            plt.close()
            plt.imshow(label, cmap='gray')
            # plt.imshow(label)
            plt.show()
            plt.close()


if __name__ == '__main__':
    read_single_dict()
