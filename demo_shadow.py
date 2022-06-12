from __future__ import division
import os, cv2
import numpy as np
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pickle
from skimage.color import rgb2lab
import sys
from shadow_removal_config import Config
sys.path.append('./matlab_imresize-master')
from imresize import imresize_matlab

print('tensor-flow-version:' + tf.__version__)
tf.compat.v1.disable_eager_execution()

def network_parameters_analysis():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable)
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim
        #print(variable_parameters)
        total_parameters += variable_parameters
    print('Number of network parameters:' + str(total_parameters))

def convert_w_to_real_w(w):
    # w was saved by applying the following
    # w_save = (2^16-1)*(w+15)/30
    # and then normalzied in the code to: w = 2*(w_save/65535-0.5)
    return 15*w

def convert_b_to_real_b(b):
    #b_save = (2^16-1) * (b+10)/20
    # and then normalzied in the code to: b = 2*(b_save/65535-0.5)
    return 10*b

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def build(input, num_channels, num_output_channels, k):
    net=slim.conv2d(input,num_channels,[k,k],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,num_channels,[k,k],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,num_channels,[k,k],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,num_channels,[k,k],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,num_channels,[k,k],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,num_channels,[k,k],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,num_channels,[k,k],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    #net=slim.conv2d(net,num_channels,[k,k],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8') #Uncomment when processing SRD
    net=slim.conv2d(net,num_channels,[k,k],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,num_output_channels,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return net

def prepare_data(shadow_train_path, shadow_test_path, ):
    train_names = []
    test_names = []

    for i in range(0, len(shadow_train_path)):
        train_names.append(os.listdir(shadow_train_path[i]))

    for i in range(0, len(shadow_test_path)):
        test_names.append(os.listdir(shadow_test_path[i]))

    return train_names, test_names

def mae_per_image(shadow_free, estimated_shadow_free, mask):
    # Calculate Mean absolute Error per image
    gt = shadow_free/255
    recovered = estimated_shadow_free/255
    mask[mask != 0] = 1
    shadow_free_lab = rgb2lab(gt)
    est_shadow_free_lab = rgb2lab(recovered)
    diff = shadow_free_lab - est_shadow_free_lab

    #Evaluate Shadow Area
    shadow_dist_map = abs(diff*np.repeat(mask[:, :, np.newaxis], 3, axis=2))
    shadow_dist = np.sum(shadow_dist_map)
    total_shadow_mask_pixels = np.sum(mask) #counting the number of pixels in 2D only

    # Evaluate Non Shadow Area
    non_shadow_mask = 1-mask
    non_shadow_dist_map = abs(diff*np.repeat(non_shadow_mask[:, :, np.newaxis], 3, axis=2))
    non_shadow_dist = np.sum(non_shadow_dist_map)
    non_shadow_shadow_mask_pixels = np.sum(non_shadow_mask)

    # Evaluate All Image
    full_img_dist = abs(diff)
    total_pic_dist = np.sum(full_img_dist)
    total_pic_mask_pixels = full_img_dist.shape[0]*full_img_dist.shape[1]

    mae_results = {
        "shadow_mae": shadow_dist/total_shadow_mask_pixels,
        "shadow_dist": shadow_dist,
        "shadow_num_pixels": total_shadow_mask_pixels,
        "full_img_mae": total_pic_dist/total_pic_mask_pixels,
        "full_img_dist": total_pic_dist,
        "full_img_num_pixels": total_pic_mask_pixels,
        "non_shadow_mae": non_shadow_dist/non_shadow_shadow_mask_pixels,
        "non_shadow_dist": non_shadow_dist,
        "non_shadow_num_pixels": non_shadow_shadow_mask_pixels
    }
    return mae_results

def inference(args, file_name, est_shadow_path, mask_path, gt_mask_path, shadow_free_path, add_mask, output_path, isDebug):
    # error - l2 diff when the image range is 0-1
    # l1_error_full_range - l1 error when the image range 0-255
    est_shadow_bgr = cv2.imread(est_shadow_path, -1)
    est_shadow_rgb = cv2.cvtColor(est_shadow_bgr, cv2.COLOR_BGR2RGB)
    if add_mask:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.expand_dims(mask, axis=2)
        shadow_with_mask = np.concatenate((est_shadow_rgb, mask), axis=2)
        mask = mask[:, :, 0] #return the mask to be 2D
        input_image = np.expand_dims(normalize_image(shadow_with_mask, args.meanPerChannel, args.stdPerChannel), axis=0)
    else:
        mask = np.zeros((est_shadow_rgb.shape[0:2]))
        input_image = np.expand_dims(normalize_image(est_shadow_rgb, args.meanPerChannel[0:3], args.stdPerChannel[0:3]), axis=0)

    if args.ignore_large_images and est_shadow_rgb.shape[1]*est_shadow_rgb.shape[2]>args.maximum_image_size:
        return -1, -1, -1

    output_image = sess.run(network, feed_dict={input: input_image})

    w_est = convert_w_to_real_w(output_image[0, :, :, 0:3])
    b_est = convert_b_to_real_b(output_image[0, :, :, 3:6])
    shadow_free_image_est_0_1 = np.clip(w_est * np.float32(est_shadow_rgb)/255 + b_est, 0, 1)
    shadow_free_image_est = np.uint8(255*shadow_free_image_est_0_1)

    shadow_free_image_bgr = cv2.imread(shadow_free_path, -1)
    shadow_free_image_rgb = cv2.cvtColor(shadow_free_image_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    shadow_free_resized = imresize_matlab(shadow_free_image_rgb, output_shape=(256, 256), method='bicubic')
    estimated_shadow_resized = imresize_matlab(shadow_free_image_est, output_shape=(256, 256), method='bicubic')
    gt_mask_resized = imresize_matlab(gt_mask, output_shape=(256, 256), method='bicubic')

    shadow_rmse, nonshadow_rmse, whole_rmse = rmse_per_image(shadow_free_resized, estimated_shadow_resized, gt_mask_resized/255)
    mae_results = mae_per_image(shadow_free_resized, estimated_shadow_resized, gt_mask_resized)

    rmse_results = {
        "shadow_rmse" :shadow_rmse,
        "nonshadow_rmse": nonshadow_rmse,
        "whole_rmse": whole_rmse
    }

    if isDebug:
        error = np.mean(np.square(shadow_free_image_est_0_1 - (np.float32(shadow_free_image_rgb)/255)))
        l1_error_full_range = np.mean(np.abs(np.float32(shadow_free_image_rgb)-np.float32(shadow_free_image_est)))

        plot_debug_image(file_name, est_shadow_rgb, mask, gt_mask, w_est, b_est, shadow_free_image_est,
                         shadow_free_image_rgb, output_path)
    else:
        error = -1
        l1_error_full_range = -1

    shadow_free_image_est = cv2.cvtColor(shadow_free_image_est, cv2.COLOR_RGB2BGR)
    return shadow_free_image_est, error, l1_error_full_range, rmse_results, mae_results

def norm_array(arr):
    # normalize the array values to be between 0-1
    out_arr = np.copy(arr)
    out_arr = out_arr - out_arr.min()
    max_arr = out_arr.max()
    if max_arr != 0:
        out_arr = out_arr/max_arr
    return out_arr

def plot_train_loss(output_path, train_loss):
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0 / float(DPI), 1080.0 / float(DPI))
    plt.plot(train_loss, label='train')

    plt.legend()
    plt.title('Loss vs Epoch, the Training loss is calculated during the back-propagation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show(block=False)
    plt.savefig(os.path.join(output_path, 'train_loss.png'), bbox_inches='tight')  # -4 to remove the file extension
    plt.close()

def plot_debug_image(file_name, shadow_image, mask, gt_mask, w_est, b_est,
                     est_no_shadow, gt_no_shadow, output_path):
    fig = plt.figure()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0 / float(DPI), 1080.0 / float(DPI))
    plt.subplot(2, 4, 1), plt.title(file_name + ' Shadow Image'), plt.imshow(shadow_image)
    plt.subplot(2, 4, 2), plt.title('Mask'), plt.imshow(mask)
    plt.subplot(2, 4, 3), plt.title('GT Mask'), plt.imshow(gt_mask)
    plt.subplot(2, 4, 4), plt.title('Shadow Free GT'), plt.imshow(gt_no_shadow)
    plt.subplot(2, 4, 5), plt.title('Estimated Shadow Free'), plt.imshow(est_no_shadow)
    plt.subplot(2, 4, 6), plt.title('Shadow free and GT diffrence')
    plt.imshow(norm_array(est_no_shadow.astype('float32')-gt_no_shadow.astype('float32')))
    norm_estimated_w = norm_array(w_est)
    plt.subplot(2, 4, 7), plt.title('predicted W'), plt.imshow(norm_estimated_w)
    norm_estimated_b = norm_array(b_est)
    plt.subplot(2, 4, 8), plt.title('predicted B'), plt.imshow(norm_estimated_b)

    filename, file_extension = os.path.splitext(file_name)
    plt.savefig(os.path.join(output_path, file_name[0:-4] + '_debug' + file_extension), bbox_inches='tight') #-4 to remove the file extension
    cv2.imwrite(os.path.join(output_path, file_name[0:-4] + '_estimated_w.png'), np.round(255*np.clip(norm_estimated_w, 0, 1)))
    cv2.imwrite(os.path.join(output_path, file_name[0:-4] + '_estimated_b.png'), np.round(255*np.clip(norm_estimated_b, 0, 1)))
    np.save(os.path.join(output_path, file_name[0:-4] + '_estimated_w.npy'), w_est)
    np.save(os.path.join(output_path, file_name[0:-4] + '_estimated_b.npy'), b_est)
    #plt.show(block=False)
    plt.close(fig)

def normalize_image(image, mean, std):
    norm_image = image.copy()
    norm_image = np.float32(norm_image)/255.0
    norm_image = norm_image-mean
    return norm_image/std

def un_normalize_image(image, mean, std):
    un_norm_image = image.copy()
    un_norm_image = un_norm_image*std
    un_norm_image = un_norm_image+mean
    return 255*un_norm_image

def un_normalize_tensor(tensor, mean, std):
    tensor = tensor*std
    tensor = tensor+mean
    return 255*tensor

def random_flip(shadow_image_aug, shadow_free_image_aug):
    # Random flip (vertical, horizontal, both):
    num = random.randint(-1, 2)
    if num != 2:
        shadow_image_aug = cv2.flip(shadow_image_aug, num)
        shadow_free_image_aug = cv2.flip(shadow_free_image_aug, num)
    return shadow_image_aug, shadow_free_image_aug

def crop_image(img, radius_to_crop):
    #img : ndarray with 4 channels, 2 element ndarray: radius_to_crop [x,y]
    return img[:, radius_to_crop[1]:-radius_to_crop[1], radius_to_crop[0]:-radius_to_crop[0], :]

def augment(shadow_image, shadow_free_image, augmentations_to_apply):
    shadow_image_aug = np.copy(shadow_image)
    shadow_free_image_aug = np.copy(shadow_free_image)
    if augmentations_to_apply["random_flip"]:
        shadow_image_aug, shadow_free_image_aug,  = random_flip(shadow_image_aug, shadow_free_image_aug)
    return shadow_image_aug, shadow_free_image_aug

def input_generator(shadow_train_path, shadow_file_name, mask_train_path, add_mask):
    shadow_train_image_bgr = cv2.imread(os.path.join(shadow_train_path, shadow_file_name), -1)
    shadow_train_image_rgb = cv2.cvtColor(shadow_train_image_bgr, cv2.COLOR_BGR2RGB)

    if add_mask:
        mask_file_path = os.path.join(mask_train_path, shadow_file_name)
        if os.path.exists(mask_file_path):
            mask_image = cv2.imread(os.path.join(mask_train_path, shadow_file_name), cv2.IMREAD_GRAYSCALE)
        else:
            print('The mask:', mask_file_path, ' was not found')
        mask_image = np.expand_dims(mask_image, axis=2)
        shadow_image_rgb_with_mask = np.concatenate((shadow_train_image_rgb, mask_image), axis=2)
        single_input = normalize_image(shadow_image_rgb_with_mask, args.meanPerChannel, args.stdPerChannel)
    else:
        single_input = normalize_image(shadow_train_image_rgb, args.meanPerChannel[0:3], args.stdPerChannel[0:3])
    return single_input

def generate_coeff_image(coeff_file_path):
    coeff_image_bgr = cv2.imread(coeff_file_path, -1)
    coeff_image_bgr = cv2.cvtColor(coeff_image_bgr, cv2.COLOR_BGR2RGB)
    coeff_image_bgr = np.float32(coeff_image_bgr) / 65535.0
    coeff_image_bgr = 2 * (coeff_image_bgr - 0.5)  # make w be between -1 and 1
    return coeff_image_bgr

def rmse_per_image(estimated_shadow, shadow_free, mask):
    # mask should be binary
    mask_repmat = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    estimated_shadow_lab = rgb2lab(estimated_shadow)
    shadow_free_lab = rgb2lab(shadow_free)
    diff = estimated_shadow_lab-shadow_free_lab
    shadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * mask_repmat).sum(axis=(0, 1)) / mask.sum())
    nonshadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * (1 - mask_repmat)).sum(axis=(0, 1)) / (1 - mask_repmat).sum())
    whole_rmse = np.sqrt(np.power(diff, 2).mean(axis=(0, 1)))
    return shadow_rmse.sum(), nonshadow_rmse.sum(), whole_rmse.sum()

def dataset_loader(args, set_names, load_training_set_to_memory, shadow_path, mask_path, shadow_free_train_path):
    # Single images loading
    all_shadow_images = []
    all_shadow_free_images = []
    num_training_datasets = len(set_names)
    if load_training_set_to_memory:
        for d in range(num_training_datasets):
            data_set_training_samples = set_names[d]
            for i in range(len(data_set_training_samples)):
                print('Loading image to memory:' + data_set_training_samples[i])
                single_input = input_generator(shadow_path[d], data_set_training_samples[i],
                                               mask_path[d], args.add_mask)
                all_shadow_images.append(single_input)
                shadow_free_image_bgr = cv2.imread(
                    os.path.join(shadow_free_train_path[d], data_set_training_samples[i]), -1)
                shadow_free_image_rgb = cv2.cvtColor(shadow_free_image_bgr, cv2.COLOR_BGR2RGB)
                shadow_free_image_rgb = normalize_image(shadow_free_image_rgb, args.meanPerChannel[0:3],
                                                        args.stdPerChannel[0:3])
                all_shadow_free_images.append(shadow_free_image_rgb)

    return all_shadow_images, all_shadow_free_images

if __name__ == "__main__":

    # datetime object containing current date and time
    dt_string = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    task = "shadow_removal_" + dt_string
    args = Config()

    if args.checkpoints_directory == "":
        output_path = os.path.join(os.getcwd(), 'Output', 'FastImageProcessing', task)
    else:
        output_path = os.path.abspath(
            os.path.join(os.path.abspath(os.path.join(args.checkpoints_directory, os.pardir)), os.pardir))
        task = os.path.basename(output_path)

    print('Output Folder:' + output_path)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    sess = tf.compat.v1.Session()

    train_mask = \
        {
            'ISTD': os.path.join(args.data_path, args.predicted_train_mask.istd),
            'SRD': os.path.join(args.data_path, args.predicted_train_mask.srd)
        }

    test_mask = \
        {
            "ISTD": os.path.join(args.data_path, args.predicted_test_mask.istd),
            'SRD': os.path.join(args.data_path, args.predicted_test_mask.srd)
        }

    shadow_train_path = []
    mask_gt_train_path = []
    mask_train_path = []
    shadow_free_train_path = []
    for i in range(0, len(args.train_data_set)):
        shadow_train_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.train_data_set[i], args.train_folder_name, args.train_shadow_folder_name))
        mask_gt_train_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.train_data_set[i], args.train_folder_name, args.train_mask_folder_name))
        mask_train_path.append(train_mask[args.train_data_set[i]])
        shadow_free_train_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.train_data_set[i], args.train_folder_name, args.train_shadow_free_folder_name))

    shadow_test_path = []
    mask_gt_test_path = []
    shadow_free_test_path = []
    mask_test_path = []
    b_gt_test_path = []
    w_gt_test_path = []
    for i in range(0, len(args.test_data_set)):
        shadow_test_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.test_data_set[i], args.test_folder_name, args.test_shadow_folder_name))
        mask_test_path.append(test_mask[args.test_data_set[i]])
        mask_gt_test_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.test_data_set[i], args.test_folder_name, args.test_mask_folder_name))
        shadow_free_test_path.append(
            os.path.join(args.data_path, args.data_set_folder, args.test_data_set[i], args.test_folder_name, args.test_shadow_free_folder_name))
    writing_results_frequency = args.num_epocs  # [in epochs]

    min_normalized_value = (0-args.meanPerChannel[0:3])/args.stdPerChannel[0:3]
    max_normalized_value = (1-args.meanPerChannel[0:3])/args.stdPerChannel[0:3]
    train_set_names, test_set_names = prepare_data(shadow_train_path, shadow_test_path)

    if args.num_images_per_set_to_load != -1:
        num_datasets = len(train_set_names)
        for i in range(num_datasets):
            if len(train_set_names[i]) > args.num_images_per_set_to_load:
                train_set_names[i] = train_set_names[i][0:args.num_images_per_set_to_load]
            if len(test_set_names[i]) > args.num_images_per_set_to_load:
                test_set_names[i] = test_set_names[i][0:args.num_images_per_set_to_load]

    input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 4]) #RGB + Mask
    shadow_free = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
    num_output_channels = 6

    coeff = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, num_output_channels])
    network=build(input, args.num_channels, num_output_channels, args.kernel_size)
    network_parameters_analysis()

    ## Loss ##
    est_shadow_free = convert_w_to_real_w(network[:, :, :, 0:3])*un_normalize_tensor(input[:, :, :, 0:3], args.meanPerChannel[0:3], args.stdPerChannel[0:3])/255 + \
                      convert_b_to_real_b(network[:, :, :, 3:6]) #image is divide by 255 since the w,b was calculated when the image was divided by 255
    est_shadow_free = tf.clip_by_value(est_shadow_free, 0, 1)
    image_se = tf.square(est_shadow_free-un_normalize_tensor(shadow_free, args.meanPerChannel[0:3], args.stdPerChannel[0:3])/255)
    loss = tf.reduce_mean(image_se)

    with tf.device("/gpu:0"):
        opt=tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss,var_list=[var for var in tf.compat.v1.trainable_variables()])

    saver=tf.compat.v1.train.Saver(max_to_keep=1000)
    sess.run(tf.compat.v1.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(args.checkpoints_directory)

    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        model_first_epoch = int(ckpt.model_checkpoint_path.split("-")[-1]) #add one to start from the next new epoch
    else:
        model_first_epoch = 0 #for untrained model

    # Located out of is_training since this script can runs at inference only, and the folders should be created
    train_folder_path = os.path.join(output_path, 'train')
    test_folder_path = os.path.join(output_path, 'test')

    if not os.path.isdir(train_folder_path):
        os.makedirs(train_folder_path)

    if not os.path.isdir(test_folder_path):
        os.makedirs(test_folder_path)

    train_epoch_path = os.path.join(train_folder_path, "%04d" % (model_first_epoch))
    test_epoch_path = os.path.join(test_folder_path, "%04d" % (model_first_epoch))

    if args.is_training and args.num_epocs>model_first_epoch:
        tbTrain = SummaryWriter(train_folder_path)
        train_loss_vs_epochs = np.zeros(args.num_epocs)
        print('Writing Train results to:\n' + train_folder_path)

        train_shadow_images, train_shadow_free_images = dataset_loader(args, train_set_names, args.load_dataset_to_memory,
                                                                                         shadow_train_path, mask_train_path, shadow_free_train_path)

        train_epoch_path = os.path.join(train_folder_path, "%04d"%(model_first_epoch))

        checkpoints_to_save_directory = os.path.join(train_folder_path, 'checkpoints')
        if not os.path.isdir(checkpoints_to_save_directory):
            os.makedirs(checkpoints_to_save_directory)

        #before entering another training loop add 1
        model_first_epoch = model_first_epoch + 1

        for epoch in range(model_first_epoch, args.num_epocs + 1):
            epoch_str = "%04d"%(epoch)
            train_epoch_path = os.path.join(train_folder_path, epoch_str)
            test_epoch_path = os.path.join(test_folder_path, epoch_str)

            print('Processing epoch:', epoch)
            if not os.path.isdir(train_epoch_path):
                total_train_loss = 0
                for data_set_id in range(len(train_set_names)):
                    cnt = 0
                    data_set_training_samples = train_set_names[data_set_id]
                    data_set_train_path = shadow_train_path[data_set_id]
                    num_training_samples = len(data_set_training_samples)
                    data_set_shadow_free_train_path = shadow_free_train_path[data_set_id]
                    for id in np.random.permutation(num_training_samples):
                        if args.load_dataset_to_memory:
                            curr_shadow_image = train_shadow_images[id]
                            curr_shadow_free_image = train_shadow_free_images[id]
                        else: #load from disk
                            curr_shadow_image = input_generator(data_set_train_path, data_set_training_samples[id],
                                                                mask_train_path[data_set_id], args.add_mask)
                            shadow_free_image_bgr = cv2.imread(os.path.join(data_set_shadow_free_train_path, data_set_training_samples[id]), -1)
                            shadow_free_image_rgb = cv2.cvtColor(shadow_free_image_bgr, cv2.COLOR_BGR2RGB)
                            curr_shadow_free_image = normalize_image(shadow_free_image_rgb, args.meanPerChannel[0:3], args.stdPerChannel[0:3])

                        # Perform augmentations
                        shadow_image_aug, shadow_free_image_aug = augment(curr_shadow_image, curr_shadow_free_image, args.augmentations_to_apply)
                        shadow_image_aug = np.expand_dims(shadow_image_aug, axis=0)
                        shadow_free_image_aug = np.expand_dims(shadow_free_image_aug, axis=0)
                        print("Processing ", data_set_training_samples[id])
                        # Due to GPU memory limitations bypass high resolution images
                        if args.ignore_large_images and shadow_image_aug.shape[1]*shadow_image_aug.shape[2]>args.maximum_image_size:
                            print('Due to GPU memory limitation skipping image size', shadow_image_aug.shape[1], shadow_image_aug.shape[2])
                            continue

                        _,current_loss=sess.run([opt,loss], feed_dict={input: shadow_image_aug,
                                                                       shadow_free: shadow_free_image_aug})
                        total_train_loss = total_train_loss + current_loss
                        cnt+=1
                        print("epoch %d dataset %d train image count %d out of %d, loss: %.6f"%(epoch, data_set_id + 1, cnt, num_training_samples, current_loss))

                normalized_train_loss = total_train_loss/num_training_samples
                train_loss_vs_epochs[epoch-1] = normalized_train_loss #the first epoch is saved as epoch 0
                print("epoch: %d total train loss: %f " % (epoch, normalized_train_loss))
                tbTrain.add_scalar("/Loss", normalized_train_loss, epoch - 1)
                saver.save(sess, os.path.join(checkpoints_to_save_directory, "model.ckpt"), global_step=epoch)

        if not os.path.isdir(train_epoch_path):
            os.makedirs(train_epoch_path)
        l1_error_full_range = -1*np.ones(len(train_set_names))

    plot_train_loss(output_path, train_loss_vs_epochs)
    # Show results on train set
    if args.evaluate_training_set:
        for data_set_id in range(len(train_set_names)):
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0

            eval_mae_shadow_dist = 0
            eval_mae_shadow_num_pixels = 0
            eval_mae_non_shadow_dist = 0
            eval_mae_non_shadow_num_pixels = 0
            eval_mae_full_img_dist = 0
            eval_mae_full_img_num_pixels = 0

            curr_train_data_set_names = train_set_names[data_set_id]
            num_curr_train_set_name = len(curr_train_data_set_names)
            for ind in range(num_curr_train_set_name):
                shadow_free_image, error, l1_error_full_range, rmse_results, mae_results = \
                    inference(args, curr_train_data_set_names[ind], os.path.join(shadow_train_path[data_set_id], curr_train_data_set_names[ind]),
                              os.path.join(mask_train_path[data_set_id], curr_train_data_set_names[ind]),
                              os.path.join(mask_gt_train_path[data_set_id], curr_train_data_set_names[ind]),
                              os.path.join(shadow_free_train_path[data_set_id], curr_train_data_set_names[ind]),
                              args.add_mask, train_epoch_path, args.isDebug)

                eval_shadow_rmse+=rmse_results["shadow_rmse"]
                eval_nonshadow_rmse+=rmse_results["nonshadow_rmse"]
                eval_rmse+=rmse_results["whole_rmse"]
                eval_mae_shadow_dist+=mae_results["shadow_dist"]
                eval_mae_shadow_num_pixels+=mae_results["shadow_num_pixels"]
                eval_mae_non_shadow_dist += mae_results["non_shadow_dist"]
                eval_mae_non_shadow_num_pixels += mae_results["non_shadow_num_pixels"]
                eval_mae_full_img_dist += mae_results["full_img_dist"]
                eval_mae_full_img_num_pixels += mae_results["full_img_num_pixels"]

                print('train:' + curr_train_data_set_names[ind] + f' loss={error}' + f' l1_error_full_range={l1_error_full_range}')
                cv2.imwrite(os.path.join(train_epoch_path, curr_train_data_set_names[ind]), shadow_free_image)

            eval_shadow_rmse = eval_shadow_rmse/num_curr_train_set_name
            eval_nonshadow_rmse = eval_nonshadow_rmse/num_curr_train_set_name
            eval_rmse = eval_rmse/num_curr_train_set_name

            eval_shadow_mae = eval_mae_shadow_dist/eval_mae_shadow_num_pixels
            eval_non_shadow_mae = eval_mae_non_shadow_dist/eval_mae_non_shadow_num_pixels
            eval_full_img_mae = eval_mae_full_img_dist/eval_mae_full_img_num_pixels

            print('Dataset:', args.train_data_set[data_set_id], 'RMSE Shadow', "{:.2f}".format(eval_shadow_rmse), 'RMSE Non Shadow', "{:.2f}".format(eval_nonshadow_rmse),
                  'All Image RMSE', "{:.2f}".format(eval_rmse), 'MAE Shadow:', "{:.2f}".format(eval_shadow_mae), "MAE Non Shadow:",
                  "{:.2f}".format(eval_non_shadow_mae), "All Image MAE:", "{:.2f}".format(eval_full_img_mae))

    if not os.path.isdir(test_epoch_path):
        os.makedirs(test_epoch_path)

    # Show results on test set
    for data_set_id in range(len(test_set_names)):
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0

        eval_mae_shadow_dist = 0
        eval_mae_shadow_num_pixels = 0
        eval_mae_non_shadow_dist = 0
        eval_mae_non_shadow_num_pixels = 0
        eval_mae_full_img_dist = 0
        eval_mae_full_img_num_pixels = 0

        curr_test_data_set_names = test_set_names[data_set_id]
        num_curr_test_set_name = len(curr_test_data_set_names)
        for ind in range(num_curr_test_set_name):
            shadow_free_image, error, l1_error_full_range, rmse_results, mae_results = \
                inference(args, curr_test_data_set_names[ind], os.path.join(shadow_test_path[data_set_id], curr_test_data_set_names[ind]),
                          os.path.join(mask_test_path[data_set_id], curr_test_data_set_names[ind]),
                          os.path.join(mask_gt_test_path[data_set_id], curr_test_data_set_names[ind]),
                          os.path.join(shadow_free_test_path[data_set_id], curr_test_data_set_names[ind]),
                          args.add_mask, test_epoch_path, args.isDebug)
            eval_shadow_rmse += rmse_results["shadow_rmse"]
            eval_nonshadow_rmse += rmse_results["nonshadow_rmse"]
            eval_rmse += rmse_results["whole_rmse"]

            eval_mae_shadow_dist += mae_results["shadow_dist"]
            eval_mae_shadow_num_pixels += mae_results["shadow_num_pixels"]
            eval_mae_non_shadow_dist += mae_results["non_shadow_dist"]
            eval_mae_non_shadow_num_pixels += mae_results["non_shadow_num_pixels"]
            eval_mae_full_img_dist += mae_results["full_img_dist"]
            eval_mae_full_img_num_pixels += mae_results["full_img_num_pixels"]

            print('test:' + curr_test_data_set_names[ind] + f' loss={error}' + f' l1_error_full_range={l1_error_full_range}')
            cv2.imwrite(os.path.join(test_epoch_path, curr_test_data_set_names[ind]), shadow_free_image)

        eval_shadow_rmse = eval_shadow_rmse/num_curr_test_set_name
        eval_nonshadow_rmse = eval_nonshadow_rmse/num_curr_test_set_name
        eval_rmse = eval_rmse/num_curr_test_set_name

        eval_shadow_mae = eval_mae_shadow_dist/eval_mae_shadow_num_pixels
        eval_non_shadow_mae = eval_mae_non_shadow_dist/eval_mae_non_shadow_num_pixels
        eval_full_img_mae = eval_mae_full_img_dist/eval_mae_full_img_num_pixels

        print('Dataset:', args.test_data_set[data_set_id], 'RMSE Shadow', "{:.2f}".format(eval_shadow_rmse), 'RMSE Non Shadow', "{:.2f}".format(eval_nonshadow_rmse),
              'All Image RMSE', "{:.2f}".format(eval_rmse), 'MAE Shadow:', "{:.2f}".format(eval_shadow_mae), "MAE Non Shadow:",
              "{:.2f}".format(eval_non_shadow_mae), "All Image MAE:", "{:.2f}".format(eval_full_img_mae))