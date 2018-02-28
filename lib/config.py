
USE_LMDB = False
IMG_LMDB_PATH = '/data/data/liuhuawei/data_lmdb_backup_for_ssd/data_lmdb_for_image_copy_and_mark_data'

METADATA_JSON = '/core1/data/home/shizhan/jiahuan/test/triplet/data/objectid_to_metadata_mix.json'
## json path to pair file, key:(objectid_a, objectid_b) val:1 or 0
PAIR_JSON = '/core1/data/home/shizhan/jiahuan/siamese_shelf/res50/training/data/test.json'
## image config
TARGET_SIZE = 224
PIXEL_MEANS = [104.0, 117.0, 123.0]

## The number of samples in each minibatch
BATCH_SIZE = 8

## prefetch process for data layer (must be false here)
USE_PREFETCH = False
RNG_SEED = 8

# BBOX_SCALE_TYPE = 'SCALE'
BBOX_SCALE_TYPE = 'ABSOLUTE'
BBOX_SCALE = 3.0
VIS = False

##### mutligpu
# METADATA_JSON_GPU = '/core1/data/home/shizhan/jiahuan/siamese_shelf/res50/training/test.json'
METADATA_JSON_GPU = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/test.json'
MAX_PAIR = 8
PN_RATIO = 2
ITER_PERIMG = 2
