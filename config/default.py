# MODE
mode = 'train'
model_name = 'kitti'

# MODEL
do_stereo = 'store_true'
encoder = 'resnet18'
decoder = 'unet'

# DATA
dataset = 'kitti'
data_path = '../../dataset/kitti/data/'
filenames_file = './data/filenames/kitti_train_files.txt'
image_path=''

# TRAIN
height = 256
width = 512

batch_size = 16
num_epochs = 50

retrain = 'store_true'

# OPTIMIZATION
learning_rate = 1e-4

# LOSS 
lr_loss_weight = 1.0
alpha_image_loss = 0.85
disp_gradient_loss_weight = 0.1

# OTHERS
wrap_mode = 'store_true'

# SYSTEM 
num_gpus = 1
num_threads = 8
full_summary = 'store_true'

# PATH
output_directory = ''
log_directory = 'models'
checkpoint_path = ''