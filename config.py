DATASET_ROOT = './'
BATCH_SIZE = 32
TEST_SIZE = 0.05

# Optimizer
# Available values:
# - 'adam'
# - 'rmsprop'
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-2

# Params. for MixedLoss
ALPHA = 10.0
GAMMA = 2.0

EPOCHS = 10

# Model
# Available values for base model:
# - 'vgg16'
# - 'resnet50v2'
BASE_MODEL = 'vgg16'
CHECKPOINTS_PATH = './'
NAME = 'vgg16_unet'

# This parameters should not be changed :(
IM_SIZE = (768, 768)
TARGET_IM_SIZE = (384, 384)