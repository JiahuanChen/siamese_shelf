import numpy as np
import lmdb
import os
import json
import cv2
import requests

import config
from data_augmentation import scale_bbox

def load_json(json_file):
    assert os.path.exists(json_file), \
            'json file not found at: {}'.format(json_file)
    with open(json_file, 'rb') as f:
        data = json.load(f)
    return data

cv_session = requests.Session()
cv_session.trust_env = False
def cv_load_image(in_):
    '''
    Return
        image: opencv format np.array. (C x H x W) in BGR np.uint8
    '''
    if in_[:4] == 'http':
        img_nparr = np.fromstring(cv_session.get(in_).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(in_)
    return img    

def get_image_blob(sample_a, sample_b):
    im_blob = []
    for i in range(config.BATCH_SIZE):
        im_a = cv_load_image(sample_a[i]['img_path'])
        im_b = cv_load_image(sample_b[i]['img_path'])
        bbox_a = [int(sample_a[i]['xmin']), int(sample_a[i]['ymin']), int(sample_a[i]['width']), int(sample_a[i]['height'])]
        bbox_a = [bbox_a[0], bbox_a[1], bbox_a[0]+bbox_a[2], bbox_a[1]+bbox_a[3]]
        bbox_b = [int(sample_b[i]['xmin']), int(sample_b[i]['ymin']), int(sample_b[i]['width']), int(sample_b[i]['height'])]
        bbox_b = [bbox_b[0], bbox_b[1], bbox_b[0]+bbox_b[2], bbox_b[1]+bbox_b[3]]
        im_a = prep_im_for_blob_with_bbox(im_a, bbox_a)
        im_b = prep_im_for_blob_with_bbox(im_b, bbox_b)
        im = np.concatenate((im_a, im_b), axis=2)
        im_blob.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(im_blob)
    return blob, np.ones((config.BATCH_SIZE, 303), dtype=np.int)     

def prep_im_for_blob_with_bbox(im, bbox):
    im_old = im.astype(np.float32, copy=False)
    bbox_old = bbox[:]
    im = im.astype(np.float32, copy=False)
    if config.BBOX_SCALE_TYPE == 'SCALE':
        bbox = np.array(bbox).reshape((1, -1))
        bbox = scale_bbox(bbox, config.BBOX_SCALE).astype(int)[0]
    elif config.BBOX_SCALE_TYPE == 'ABSOLUTE':
        bbox[0] -= int(config.BBOX_SCALE)
        bbox[1] -= int(config.BBOX_SCALE)
        bbox[2] += int(config.BBOX_SCALE)
        bbox[3] += int(config.BBOX_SCALE)
    else:
        raise

    H, W, _ = im.shape
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = min(bbox[2], W)
    bbox[3] = min(bbox[3], H)
    
    im = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    target_size = config.TARGET_SIZE
    pixel_means = np.array([[config.PIXEL_MEANS]])
    if im.shape[0]*im.shape[1] == 0:
        print im_old.shape, im.shape
        print bbox_old, bbox
    im = cv2.resize(im, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    im -= pixel_means
    return im      

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    target_size = config.TARGET_SIZE
    pixel_means = np.array([[config.PIXEL_MEANS]])   
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im = cv2.resize(im, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    return im

class SampleData(object):
    def __init__(self):
        if not config.USE_LMDB:
            assert os.path.exists(config.METADATA_JSON), config.METADATA_JSON
            self._objectid_to_meta = load_json(config.METADATA_JSON)
        self._pairs = load_json(config.PAIR_JSON)
        self._keys = self._pairs.keys()
        print 'Num of training datas: ', len(self._keys) 

class SampleData_Multi_GPU(object):
    def __init__(self):
        # JSON format: 
        # {'imgid_a,imgid_b':{'path_a':'', 'path_b':'' , bbox_a:[[],[],...], bbox_b[[],[],...]}, 'imgid_a,imgid_b'}
        # all values are exactly the same as those in sql
        self._imgid_pairs = load_json(config.METADATA_JSON_GPU)
        self._keys = self._imgid_pairs.keys()
        print 'Num of training datas: ', len(self._imgid_pairs) 

class FeatureLmdb(object):
    def __init__(self, db, **kwargs):
        self.env = lmdb.open(db, map_size=1e12)

    def put(self, keys, feas, type_='fea'):
        '''Store features in to lmdb backend
        Args:
            keys(list): keys for features
            feas(list): corresponding features
            type_(str): feature type, fea for numpy.ndarray, raw for serialized features
        '''
        with self.env.begin(write=True) as txn:
            for i, k in enumerate(keys):
                if type_ == 'fea':
                    txn.put(k, feas[i].tobytes())
                elif type_ == 'raw':
                    txn.put(k, feas[i])
                else:
                    raise
    def get(self, key):
        '''
        Get feature by key from the lmdb.
        Args:
            key(str): key of the feature
        '''
        with self.env.begin() as txn:
            fea = np.fromstring(txn.get(key), dtype=np.float32)
        return fea

    def get_all(self):
        '''
        Get all features from the lmdb.
        '''
        all_data = {}
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                all_data[k] = np.fromstring(v, dtype=np.float32)
        return all_data

    def get_raws(self):
        '''
        Get all serialized features from the lmdb
        '''
        keys = []
        values = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                keys.append(k)
                values.append(v)
        return keys, values                                        

if __name__ == '__main__':
    sample = sampledata()
