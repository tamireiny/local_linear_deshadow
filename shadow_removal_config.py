import os
import numpy as np

class dataset:
    def __init__(self, istd, srd):
        self.istd = istd
        self.srd = srd

class Config:
    def __init__(self,
        data_path = r"F:\Dropbox\TAU\Deep Learning\ShadowRemoval",
        augmentations_to_apply = {'random_flip': True},
        run_print_network_w0_w1_weights= False,
        add_mask= True,
        num_channels = 32, #CAN24 default is 24 channels per layer
        kernel_size = 3,
        meanPerChannel = np.array([0.485, 0.456, 0.406, 0.0]),
        stdPerChannel = np.array([0.229, 0.224, 0.225, 1.0]),
        load_dataset_to_memory = True,
        isDebug= True,
        ignore_large_images=True,
        maximum_image_size=736*896+1, #840*640, #To avoid of GPU out of memory
        # To load checkpoint, set the path to the checkpoint folder path, if loading is not required, set ''
        checkpoints_directory= "",
        is_training=True,
        evaluate_training_set=True,
        data_set_folder = 'Datasets',
        train_folder_name='train',
        test_folder_name='test',
        train_data_set= ['ISTD'], #['SRD'], #list can be 'ISTD' or "SRD'
        test_data_set= ['ISTD'], #['SRD'], #
        train_shadow_folder_name= 'train_A',
        train_mask_folder_name= 'train_B',
        train_shadow_free_folder_name= 'train_C',
        test_shadow_folder_name = 'test_A',
        test_mask_folder_name= 'test_B',
        test_shadow_free_folder_name= 'test_C',
        use_gt_mask=False,
        predicted_train_mask = dataset(istd=os.path.join('Output', 'BDRAR', 'ISTD', 'Input_ISTD_output_ISTD_Train_Binary'),
                                       srd=os.path.join('Models', 'BDRAR', 'ckpt', 'SRD_LR_5e-4_if_val_dont_improve_in_2e-3_in_10_epochs',
                                                    'best_model_epoch_9_output_srd_train_resized_bicubic')),
        predicted_test_mask = dataset(istd=os.path.join('Output', 'BDRAR', 'ISTD', 'Input_ISTD_output_ISTD_Test_Binary'),
                                      srd=os.path.join('Models', 'BDRAR', 'ckpt', 'SRD_LR_5e-4_if_val_dont_improve_in_2e-3_in_10_epochs','best_model_epoch_9_output_srd_test_resized_bicubic')),
        num_epocs = 1,
        learning_rate= 0.0001,
        num_images_per_set_to_load = -1 #set -1 to process all images
    ):
        self.data_path = data_path
        self.augmentations_to_apply = augmentations_to_apply
        self.run_print_network_w0_w1_weights = run_print_network_w0_w1_weights
        self.add_mask = add_mask
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.meanPerChannel = meanPerChannel
        self.stdPerChannel = stdPerChannel
        self.load_dataset_to_memory = load_dataset_to_memory
        self.isDebug = isDebug
        self.ignore_large_images = ignore_large_images
        self.maximum_image_size = maximum_image_size
        self.checkpoints_directory = checkpoints_directory
        self.is_training = is_training
        self.evaluate_training_set = evaluate_training_set
        self.data_set_folder = data_set_folder
        self.train_folder_name = train_folder_name
        self.test_folder_name = test_folder_name
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set
        self.train_shadow_folder_name = train_shadow_folder_name
        self.train_mask_folder_name = train_mask_folder_name
        self.train_shadow_free_folder_name = train_shadow_free_folder_name
        self.test_shadow_folder_name = test_shadow_folder_name
        self.test_mask_folder_name = test_mask_folder_name
        self.test_shadow_free_folder_name = test_shadow_free_folder_name
        self.use_gt_mask = use_gt_mask
        self.predicted_train_mask = predicted_train_mask
        self.predicted_test_mask = predicted_test_mask
        self.num_epocs = num_epocs
        self.learning_rate = learning_rate
        self.num_images_per_set_to_load = num_images_per_set_to_load