"""Train the network."""
import _init_paths
import time
import numpy as np
import sys
sys.path.insert(0, '/core1/data/home/shizhan/jiahuan/siamese_shelf/res50/training/lib')
from multiprocessing import Process, Queue

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format
import google.protobuf as pb2
from timer import Timer
import config
from utils import SampleData, get_image_blob

import argparse
import sys
sys.path.insert(0, '/core1/data/home/shizhan/jiahuan/test/triplet')
from show_pairs import init_mysql
from mysql_handle import MySql
mysql = init_mysql()
import matplotlib.pyplot as plt

# import utils
# print utils.__file__
# exit()

class DataProviderTask(object):
    def __init__(self, data_container):
        super(DataProviderTask, self).__init__()  
        self._data_container = data_container
        self._index = 0
        np.random.shuffle(self._data_container._keys)

        self._pair_num = config.BATCH_SIZE 

    def _get_next_minibatch(self):
        pair_num = self._pair_num

        sample_a = []
        sample_b = []
        sample_label = []
        cnt = 0
        while  cnt < pair_num:
            if self._index >= len(self._data_container._keys):
                self._index = 0
                np.random.shuffle(self._data_container._keys)
            a_b = self._data_container._keys[self._index]   
            label = self._data_container._pairs[a_b]
            a = a_b.strip().split(',')[0]
            b = a_b.strip().split(',')[1]
            sample_a.append(a)
            sample_b.append(b)
            sample_label.append(label)              
            self._index +=1 
            cnt += 1       
        sample_a = [self._data_container._objectid_to_meta[object_id] for object_id in sample_a] 
        sample_b = [self._data_container._objectid_to_meta[object_id] for object_id in sample_b] 
        
        return (sample_a, sample_b, sample_label)
           
class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt,
                 pretrained_model=None, gpu_id=0):
        """Initialize the SolverWrapper."""
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def train_model(self, max_iters, queue):
        """Network training loop."""
        timer = Timer()
        read_time = 0.0
        cnt = 0
        while self.solver.iter < max_iters:
            timer.tic()
            s_time = time.time()
            blobs = queue.get() 
            train_X = blobs['data'].astype(np.float32)
            train_Y = blobs['labels'].astype(np.float32)

            #### check training images
            if config.VIS == True:
                print len(train_X)
                print train_X.shape
                mean = np.array([1.040069879317889e+02,1.166687676169677e+02,1.226789143406786e+02])
                for i in range( config.BATCH_SIZE ):
                    img = train_X[i]
                    img_a = img[:3,:,:]
                    img_a = img_a.transpose(1,2,0)
                    img_a += mean
                    img_a/=255
                    img_a = img_a[:,:,(2,1,0)]
                    plt.subplot(1,2,1)
                    plt.imshow(img_a)

                    img_b = img[3:,:,:]
                    img_b = img_b.transpose(1,2,0)
                    img_b += mean
                    img_b/=255
                    img_b = img_b[:,:,(2,1,0)]
                    plt.subplot(1,2,2)
                    plt.imshow(img_b)
                    
                    plt.show()

            e_time = time.time()
            read_time += (e_time - s_time)
            cnt += 1
            self.solver.net.set_input_arrays(train_X, train_Y)            
            self.solver.step(1)	    
            # print 'conv5_3:',self.solver.net.params['conv_stage3_block2_branch2c'][0].data[0][0][0]
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)   
                print 'Batch reading time is {:.3f}s / batch'.format(read_time / cnt)
                read_time = 0.0
                cnt = 0       
        self.solver.snapshot()

def write_worker(q_out, solver_prototxt, pretrained_model, gpu_id):
    # 
    sw = SolverWrapper(solver_prototxt, pretrained_model, gpu_id) 
    max_iters = sw.solver_param.max_iter 
    print 'Solving...'
    sw.train_model(max_iters, q_out)
    print 'done solving'    
     
def read_worker(q_in, q_out):
    backoff = 0.1
    cnt = 0
    while True:
        deq = q_in.get()
        if deq is None:
            break
        sample_a, sample_b, sample_label = deq      
        try:
            im_blob, _ = get_image_blob(sample_a, sample_b)
        except Exception, e:
            print e
            print 'bad data: ' # sample_a, sample_b
            continue 
        labels_blob = np.array(sample_label,dtype='float32')   
        blobs = {'data': im_blob,
         'labels': labels_blob}
        # q_out.put(blobs) 
        if q_out.qsize() < 40: 
            q_out.put(blobs) 
            backoff = 0.1
        else:
            # print 'QUEUE!!!!!!!!!!!!!!!!!!!!!!!!!!'
            q_out.put(blobs)
            time.sleep(backoff)
            backoff *= 2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a triplet network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use eg. 0',
                        default=0, type=int)
    parser.add_argument('--process', dest='num_process',
                        help='number of processes to read data',
                        default=5, type=int)    
    parser.add_argument('--solver', dest='solver_prototxt',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    """Train network.

    e.g python  tools/train_siamese.py \
    --gpu 0 --process 4 --solver solver_multiprocess.prototxt \
    --weight ../models/siamese_res50.caffemodel 

    ps: 
    1. batch_size for MemoryData Layer should be the same as config.BATCH_SIZE
    """
    args = parse_args()
    solver_prototxt = args.solver_prototxt
    pretrained_model = args.pretrained_model
    gpu_id = args.gpu_id
    number_of_processes = args.num_process    

    q_out = Queue()
    q_in = [Queue(10) for i in range(number_of_processes)]
    workers = [Process(target=read_worker, args=(q_in[i], q_out)) for i in xrange(number_of_processes)]
    for w in workers:
        w.daemon = True
        w.start() 

    write_process = Process(target=write_worker, args=(q_out, solver_prototxt, pretrained_model, gpu_id))
    write_process.daemon = True
    write_process.start()   

    data_container = SampleData()
    data_prov = DataProviderTask(data_container)

    queue_ind = 0    
    while True:
        sample = data_prov._get_next_minibatch()   
        q_in[queue_ind%number_of_processes].put(sample) 
        queue_ind += 1





