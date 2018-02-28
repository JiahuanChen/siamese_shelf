# --------------------------------------------------------
# TRIPLET LOSS
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
"""
import caffe
import numpy as np
import yaml
import config


class SiameseLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""
        layer_params = yaml.load(self.param_str)
        self.margin = layer_params['margin']
        self._iter = 0
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        loss = 0.0

        ## n*c
        a_batch = bottom[0].data[:]
        b_batch = bottom[1].data[:]
        n = a_batch.shape[0]

        self.dist_sqr = np.zeros( (n,n) , dtype='float32' )
        loss_y1 = 0
        loss_y0 = 0
        for i in range(n):
            for j in range(n):
                a = a_batch[i]
                b = b_batch[j]
                # a_norm = np.linalg.norm(a)
                # b_norm = np.linalg.norm(b)
                # a = a/a_norm if a_norm != 0 else a*0
                # b = b/b_norm if b_norm != 0 else b*0
                d2 = np.sum( (a-b) **2, axis=0, keepdims=False)
                # if i == j:
                #     print 1,d2
                # else:
                #     print 2,d2
                self.dist_sqr[i][j] = d2
                if j == i : ## similer label is 1
                    l = d2
                    loss_y1 += l
                    # print  self._iter, '   ', d2
                else:
                    l = max(self.margin-np.sqrt(d2), 0)**2
                    loss_y0 += l
                loss += l
        d2_y1 = 0
        for i in range(n):
            d2_y1 += self.dist_sqr[i,i]
        d2_y1 /= n
        d2_y0 = np.sum(self.dist_sqr) - d2_y1
        d2_y0 /= (n*n-n)
        print d2_y1,d2_y0, loss_y1/(n*n*2), loss_y/(n*n*2)
        ### l = 1/(2N)*(sigma(d2+max(...)))
        loss /= n*n*2
        # print loss
        top[0].data[...] = loss
        self._iter += 1

        ##n
        # a_b = np.sum((a_batch-b_batch)**2, axis=1, keepdims=False)
        # a_n = np.sum((a_batch-n_batch)**2, axis=1, keepdims=False)
        # dist = self.margin + a_p - a_n
        # self.dist = dist  ##cache for backward

        # ## statisticals
        # hard_triplet = np.sum(a_p>=a_n)
        # no_loss_triplet = np.sum(dist<=0.0)
        # semi_hard_triplet = self.triplet - hard_triplet - no_loss_triplet
        # print 'Semi-hard Batch(effective):'+str(semi_hard_triplet)+' Hard triplet:'+\
        #         str(hard_triplet)+' No loss triplet:'+str(no_loss_triplet)

        # loss = np.sum(dist[dist>0]) / (2.0*self.triplet)
        # top[0].data[...] = loss
    
    def backward(self, top, propagate_down, bottom):
        assert len(propagate_down) == 2, 'bottom shape should be 2 !!!'
        a_batch = bottom[0].data[:]
        b_batch = bottom[1].data[:]
        # print a_batch 
        
        a_diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        b_diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        
        # print self.dist_sqr
        # print diff.shape
        n = config.BATCH_SIZE
        for i in range(n):
            for j in range(n):
                a = a_batch[i]
                b = b_batch[j]
                # a_norm = np.linalg.norm(a)
                # b_norm = np.linalg.norm(b)
                # a = a/a_norm if a_norm != 0 else a*0
                # b = b/b_norm if b_norm != 0 else b*0
                if i == j:
                    ## y == 1, l = d^2
                    g = 2*(a - b)
                else:
                    ### y == 0, l = max(margin - d, 0)^2
                    d = np.sqrt(self.dist_sqr[i][j])
                    # print d
                    if self.margin - d > 0:
                        if d == 0:
                            g = 0
                            print '!!!!!'
                        else:
                            g = 2*(d-1)*(a-b)/d
                    else:
                        g = 0
                a_diff[i] += g
                b_diff[j] -= g
        a_diff /= n*n
        b_diff /= n*n
        bottom[0].diff[...] = a_diff
        bottom[1].diff[...] = b_diff
        
        # print np.sum(diff)


        # if propagate_down[0]:
        #     diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        #     a_batch = bottom[0].data[:self.triplet]
        #     p_batch = bottom[0].data[self.triplet:2*self.triplet]
        #     n_batch = bottom[0].data[2*self.triplet:3*self.triplet]

        #     idx = np.where(self.dist>0)[0]
        #     # backward for anchor
        #     diff[idx] = self.a * (n_batch[idx]-p_batch[idx]) / self.triplet
        #     # backward for positive
        #     diff[idx+self.triplet] = self.a * (p_batch[idx]-a_batch[idx]) / self.triplet
        #     # bakcward for negative
        #     diff[idx+2*self.triplet] = self.a * (a_batch[idx]-n_batch[idx]) / self.triplet

        #     bottom[0].diff[...] = diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass 
