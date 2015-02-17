import sys
import time

import os
import theano
import theano.tensor as T
from theano import shared
import numpy as np
import cPickle

DO_IMAGES=False
if DO_IMAGES:
    from tools import image_tiler
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

# Constants and helpers
floatX = 'float32'
constant = lambda x: theano.tensor.constant(np.asarray(x, dtype=floatX))


def log_sum_exp(x, axis):
    """ Trick for numerically stable log of sum of exps """
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))


def inlineprint(*args):
    msg = '\r' + ' '.join([str(a) for a in args])
    sys.stdout.write(msg)
    sys.stdout.flush()


class NADEk(object):
    def __init__(self, config, data):
        print 'Init ', config.model.name
        self.rng_numpy = np.random.RandomState(config.random_seed)
        self.rng_theano = MRG_RandomStreams(config.random_seed)
        self.signature = config.dataset.signature
        #   signature is basically the name of the
        #   dataset. 
        #   e.g. MNIST or caltech_silhouettes
        self.save_path = config.save_path
        self.config = config
        self.model = config.model
        #   model contains the params of the model
        #   i.e. name, n_in,n_out, n_hidden,n_layers
        #   hidden_act , tied_weights(bool), ...
        self.state = config.state
        #   Initially config.state is an empty DD()
        self.global_config = config
            #NOTE: global_config doesn't seem to 
            #   be mentioned or used anywhere after
            #   the line above! 
        # number of hidden layers besides the first and last

        config_train = config.model.train
        # dd() that contains the parameters for training
        self.valid_freq = config_train.valid_freq
        self.n_orderings = config_train.n_orderings
        self.sgd_type = config_train.sgd_type
        self.n_epochs = config_train.n_epochs
        self.minibatch_size = config_train.minibatch_size
        self.lr = config_train.lr  # learning rate
        # number of variationa inference to do
        self.k = config_train.k
        self.verbose = config_train.verbose
        self.fine_tune_n_epochs = config_train.fine_tune.n_epochs
        self.fine_tune_activate = config_train.fine_tune.activate
        assert self.model.n_layers >= 1
        self.cost_from_last = config.model.cost_from_last
        #self.momentum = config.model.train.momentum
        # NOTE: It seems like adadelta and momentum 
        #   are exclusive. So if you switch between the
        #   two, comment out the appropriate lines in 
        #   both this file and the config file
        self.adadelta_epsilon = config.model.train.adadelta_epsilon
        self.sampling_fn = None
        #fn = function. TODO: Figure out what
        #   kind of function it is and what it's purpose is

        # used in training, save also to txt file.
        self.LL_valid = []

        # for dataset
        self.trnset, self.valset, self.tstset = data
        self.marginal = np.mean(np.concatenate(
            [self.trnset, self.valset], axis=0), axis=0)
        #NOTE: This,the mean of the training and validation
        #      pixels (possibly) represents a prior 
        #       for the marginal.TODO: verify veracity

        # for tracking the costs for both pretrain and finetune
        self.costs = []
        self.costs_steps = []

        # decrease learning rate
        #self.lr_decrease = self.lr / self.n_epochs
        #NOTE: We're probably commenting the lines above
        #   because we're using adadelta. 

        acts = {
            'sigmoid': T.nnet.sigmoid,
            'tanh': T.tanh,
            'relu': lambda x: T.maximum(0., x),
            'linear': lambda x: x,
        }
        self.activation = acts[self.model.hidden_act]

    def split_into_batches(self, data, mb_size):
        if 'MNIST' in self.signature:
            n_minibatches = data.shape[0] // mb_size
            batches = np.array_split(data, n_minibatches)
        elif 'silhouettes' in self.signature:
            n_minibatches = data.shape[0] // mb_size
            batches = np.array_split(data, n_minibatches)
        else:
            raise Exception('Unknown signature: ' + self.signature)
        return batches

    def generate_masks(self, shape):
        # to generate masks for deep orderless nade training
        """
        Returns a random binary maks with ones_per_columns[i] ones
        on the i-th column

        shape: (minibatch_size * n_dim)
        Example: random_binary_maks((3,5),[1,2,3,1,2])
        Out:
        array([[ 0.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  1.]])
        """
        assert(len(shape) == 2)
        ones_per_column = self.rng_numpy.randint(shape[1], size=shape[0])
        shape_ = shape[::-1]
        indexes = np.asarray(range(shape_[0]))
        mask = np.zeros(shape_, dtype="float32")
        for i, d in enumerate(ones_per_column):
            self.rng_numpy.shuffle(indexes)
            mask[indexes[:d], i] = 1.0
        return mask.T

    def init_params(self):
        """
        Here, the params in the function name indicated the 
        params for the theano 'updates' param. In our specific case, 
        it's the weight and biase matrices/vectors
        """
        size = (self.model.n_in, self.model.n_hidden1)
        self.W1 = shared(self.weight_init(size, self.model.init_weights),
                         name='W1')
        self.Wflags = shared(self.weight_init(size, self.model.init_weights), name='Wflags')
        self.b1 = shared(self.weight_init(self.model.n_hidden1, 'zeros'), name='b_1')
        self.c = shared(self.weight_init(self.model.n_in, 'zeros'), name='c')
        self.c.set_value(-np.log((1-self.marginal) / self.marginal).astype(floatX))

        self.params = [self.W1, self.Wflags, self.b1, self.c]
        #TODO: When modifying this file for the third architecture, modify
        #       the print statements below to reflect what matrices are tied 
        #       and what aren't
        if self.model.tied_weights:
            print 'W1 and V are tied'
            self.V = self.W1
        else:
            print 'W1 and V are untied'
            self.V = shared(self.weight_init(size, self.model.init_weights),
                            name='V')
            self.params += [self.V]

        ##########
        #--- Arch 3 specific code
        #########
        # TODO: Figure out whether both the hidden 'layers' have the same number
        #       of hidden units. The same doubt holds with architecture2 

        self.W2 = shared(self.weight_init((self.model.n_hidden1, 
                                           self.model.n_hidden2),
                                           self.model.init_weights),
                                           name = 'W2')
        # NOTE: V2 was added so that all 4 weight matrices can be untied 
        #       and independant of each other. Also, if V2 is being used. 
        #       ensure that it's added to self.params by uncommenting the line after the fall
        self.V2 = shared(self.weight_init((self.model.n_hidden2, 
                                           self.model.n_hidden1),
                                           self.model.init_weights),
                                           name = 'V2')
        self.b2 = shared(self.weight_init(self.model.n_hidden2,'zeros'),name='b2')
        self.params += [self.W2,self.V2,self.b2]
 #       self.params += [self.W2,self.b2]



        ############
        #- This code is just from the original implementaiton
        #- TODO: Figure out what to do with it - keep / remove
        ############
        #TODO: Modify the next few lines below to accommodate 
        #       n_layers>2
#        if self.model.n_layers == 2:
#            self.W2 = shared(self.weight_init((self.model.n_hidden,
#                                               self.model.n_hidden),
#                                              self.model.init_weights),
#                             name='W2')
#            self.b2 = shared(self.weight_init(self.model.n_hidden, 'zeros'), 'b_2')
#            self.params += [self.W2, self.b2]

        if self.model.init_mean_field_beta:
            self.betas = shared(self.weight_init(self.model.n_in, 'ones'), 'betas')
            self.params += [self.betas]

        if self.model.train.beta_train.active:
            self.final_betas = shared(self.weight_init(self.model.n_in, 'ones'), 'final_betas')

    def compile(self, cost_from_last):
        x = T.fmatrix('inputs')
        mask = T.fmatrix('masks')
        t = self.trnset[:self.minibatch_size]
        x.tag.test_value = t
        mask.tag.test_value = self.generate_masks(t.shape)
        #The var.tag is used during theano function compilation
        #   for debugging - and is mostly used to ensure
        #   that the parameters passed are of compatible shapes

        # COST FUNCTION
        print 'Building cost function...'
        output_mask = constant(1) - mask
        D = constant(self.model.n_in) #The size of the input image/vector
        d = mask.sum(1) #d is a vector containing the number of missing bits/pixels per image
                        # in the training batch
        cost = constant(0)
        print 'Do %d steps of mean field inference' % self.k
        P = self.get_nade_k_mean_field(x, mask, self.k, use_noise=True)
        costs = []
        for i, p in enumerate(P):
            #NOTE: 'i' doesn't seem to be used anywhere. 
            # Loglikelihood on missing bits
            lp = ((x*T.log(p) + (constant(1)-x)*T.log(constant(1)-p))
                  * output_mask).sum(1) * D / (D-d)
            #TODO: 
            #   The paper uses D/(D-d+1) instead of the above formulation. 
            #   we speculate that this has something to do with the 'bias' 
            #   of the estimator. 
            #   We could, if time permits, verify the difference that the +1 makes. 
            this_cost = -T.mean(lp)
            #TODO:
            #   the line above seems redundant. We're already performing
            #   a sum in the line where 'lp' is computed. So it doesn't make
            #   sense to calculate the mean of a scalar. Effectively, 
            #   what this lines seems to be doing is computing the negative of 'lp'
            #   We'd have to check what effect commenting the line out has. 
            costs.append(this_cost)
        costs_by_step = T.stack(costs)
        if not cost_from_last:
            # If you're not using 'cost_from_last', then the cost
            #   is the average of the costs for each mean field iteration
            # Else, it is just the cost from the last mean field iteration
            cost = T.mean(T.stacklists(costs))
        else:
            cost = costs[-1]

        if self.model.train.l2:
            for param in self.params:
                if param.ndim == 2:
                    #So we're only using the weights of the connections
                    #   for computing the contribution of the regularization term. 
                    #(as opposed to also using the weights of the bias terms). 
                    cost += T.sum(param**2) * constant(self.model.train.l2)

        self.learning_rate = shared(np.float32(self.lr), name='learning_rate')

        # GRADIENTS
        grads = T.grad(cost, self.params)

        if self.sgd_type == 'adadelta':
            print 'Using adadelta SGD'
            updates = self.adadelta(grads, self.params,
                                    self.model.train.adadelta_epsilon,
                                    decay=0.95)
        else:
            raise Exception('Unknown SGD type: ' + self.sgd_type)

        # compile training functions
        print 'Compiling...'
        self.train_fn = theano.function(
            inputs=(x, mask),
            outputs=[cost, costs_by_step],
            updates=updates,
            name='train_fn'
        )

        self.sampling_fn = self.compile_sampling_fn(self.k)
        # this is build later
        self.ll_est_fn = None
        self.inpainting_fn = None

    def adadelta(self, grads, params, epsilon, decay=0.95):
        decay = constant(decay)
        epsilon = constant(epsilon)
        res = OrderedDict()
        for param, grad in zip(params, grads):
            # Mean of squared grad := E[g^2]_{t-1}
            E_sqr_grad = shared(np.zeros(param.get_value().shape, dtype=floatX))
            # Mean of squared x delta := E[(\Delta x)^2]_{t-1}
            E_sqr_dx = shared(np.zeros(param.get_value().shape, dtype=floatX))
            #TODO: 
            #   add the 'borrow = True' flag for both the get_value() calls
            #   made above. The implementation above creates deep copies when
            #   making the calls
            if param.name is not None:
                E_sqr_grad.name = 'E_sqr_grad_' + param.name
                E_sqr_dx.name = 'E_sqr_dx_' + param.name

            # Accumulate gradient
            new_E_sqrd_grad = decay * E_sqr_grad + (1 - decay) * T.sqr(grad)
            res[E_sqr_grad] = new_E_sqrd_grad

            # Compute update
            rms_dx_tm1 = T.sqrt(E_sqr_dx + epsilon)
            rms_grad_t = T.sqrt(new_E_sqrd_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

            # Accumulate updates
            new_E_sqr_dx = decay * E_sqr_dx + (1 - decay) * T.sqr(delta_x_t)
            res[E_sqr_dx] = new_E_sqr_dx 
            
            #Apply update
            res[param] = param + delta_x_t
        return res

    def compile_sampling_fn(self, k, collect_mean=False):
        # give one sample from NADE-k
        # this is a not so-optimized version, running the full model each time
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        samples_init = constant(np.zeros((self.model.n_in,)))
        means_init = constant(np.zeros((self.model.n_in,)))
        # [0,1,0,0,1,0] where 1 indicates bits that are observed
        input_mask_init = constant(np.zeros((self.model.n_in,)))
        def sample_one_bit(
                this_bit,     # the idx in the ordering that is sampled this time
                sampled,      # [x1, 0, 0, x4, 0, 0, x7] with some bits already sampled
                meaned,
                input_mask):  # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
            one = constant(1)
            # [0,0,0,1,0,0,0] where 1 indicates bits that mean field is trying to predict
            output_mask = T.zeros_like(input_mask)
            output_mask = T.set_subtensor(output_mask[this_bit], one)
            means = self.get_nade_k_mean_field(sampled, input_mask, k)
            # use the mean coming from the last step of mean field
            use_mean = means[-1]
            bit = self.rng_theano.binomial(p=use_mean,n=1,size=use_mean.shape,dtype=floatX)
            new_sample = sampled * input_mask + output_mask * bit
            new_mean = meaned * input_mask + output_mask * use_mean
            # set the new input mask
            input_mask = T.set_subtensor(input_mask[this_bit], one)
            return new_sample, new_mean, input_mask

        [samples, means, input_mask], updates = theano.scan(
            fn=sample_one_bit,
            outputs_info=[samples_init, means_init, input_mask_init],
            sequences=ordering,
        )

        out = means[-1] if collect_mean else samples[-1]
        return theano.function(
            inputs=[ordering],
            outputs=out,
            updates=updates, name='nade_k_sampling_fn')

    def compile_ll_est_ensemble(self, k, n_orderings):
        # TODO: 
        #   The code below assumes that only matrix W (or in our notation, W1)
        #   is fixed. But, in architecture 2, should W2 also be fixed and
        #   only 'V' varied between 'ensembles'? What about architecture 3? 
        #   should we use a different matrix instead of W^T?

        # 1/M sum_M log (sum_K 1/k p(x_m | o_k))
        #
        ordering = T.imatrix('ordering')
        # (O,D)
        ordering.tag.test_value = np.repeat(
            np.arange(self.model.n_in)[np.newaxis, :], n_orderings, axis=0).astype('int32')
        # (O,D)
        input_mask_init = constant(np.zeros((n_orderings, self.model.n_in)))
        x = T.fmatrix('samples')
        x.tag.test_value = self.input_test_value()
        x_ = x.dimshuffle(0, 'x', 1)

        def compute_LL_one_column(
            this_bit_vector,   # vector
            input_mask, # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
            x, x_, W1, Wflags, c):
            one = constant(1)
            means = self.get_nade_k_mean_field(x_, input_mask, k)
            # use the mean coming from the last step of mean field
            # (M,O,D)
            use_mean = means[-1]
            # (M,O)
            use_mean_shape = use_mean.shape
            use_mean = use_mean.reshape([use_mean_shape[0], use_mean_shape[1] * use_mean_shape[2]])

            idx = use_mean_shape[2] * T.arange(use_mean_shape[1]) + this_bit_vector

            mean_column = use_mean[:, idx] * constant(0.9999) + constant(0.0001 * 0.5)

            x_column = x_.reshape([x_.shape[0], x_.shape[2]])[:, this_bit_vector]

            # (M,O)
            LL = x_column*T.log(mean_column) + \
                (constant(1)-x_column)*T.log(constant(1)-mean_column)
            # set the new input mask: (O,D)

            input_mask_shape = input_mask.shape
            input_mask = input_mask.flatten()
            idx = input_mask_shape[1] * T.arange(input_mask_shape[0]) + this_bit_vector
            input_mask = T.set_subtensor(input_mask[idx], one)
            input_mask = input_mask.reshape(input_mask_shape)
            return LL, input_mask

        [LLs, input_mask], updates = theano.scan(
            fn=compute_LL_one_column,
            outputs_info=[None, input_mask_init],
            sequences=[ordering.T],
            non_sequences=[x, x_, self.W1, self.Wflags, self.c],
        )
        # LLs: (D,M,O)
        LL = log_sum_exp(LLs.sum(axis=0), axis=-1) - T.log(ordering.shape[0])
        LL_orders = LLs.sum(axis=0)
        f = theano.function(
            inputs=[x, ordering],
            outputs=[LL, LL_orders],
            updates=updates, name='LL_on_one_example_fn'
        )
        return f

    def compile_ll_est(self, k):
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        # [0,1,0,0,1,0] where 1 indicates bits that are observed
        input_mask_init = constant(np.zeros((self.model.n_in,)))
        x = T.fmatrix('samples')
        x.tag.test_value = self.input_test_value()

        def compute_LL_one_column(col, input_mask, x):
            """ col is the column currently being calculated
                input_mask indicates which bits have been sampled
                x is test vector
            """
            one = constant(1)
            # x is (D,B)
            means = self.get_nade_k_mean_field(x.T, input_mask, k)
            # use the mean coming from the last step of mean field
            mean_column = means[-1][:, col] * constant(0.9999) + constant(0.0001 * 0.5)
            x_column = x[col, :]
            LL = x_column * T.log(mean_column) + \
                (one - x_column) * T.log(one - mean_column)
            # set the new input mask
            input_mask = T.set_subtensor(input_mask[col], one)
            return LL, input_mask

        # LLs (D,B)
        [LLs, input_mask], updates = theano.scan(
            fn=compute_LL_one_column,
            sequences=[ordering],
            outputs_info=[None, input_mask_init],
            non_sequences=[x],
        )
        log_likelihood = LLs.sum(axis=0)
        f = theano.function(
            inputs=[x, ordering],
            outputs=log_likelihood,
            updates=updates, name='nade_k_LL_fn'
        )
        return f

    def input_test_value(self, n=None):
        n = self.minibatch_size if n is None else n
        return np.random.binomial(
            n=1, p=0.5,
            size=(n, self.model.n_in)).astype(floatX)

    def weight_init(self, size, type):
        if type == 'zeros':
            value = np.zeros(size)
        elif type == 'ones':
            value = np.ones(size)
        elif type == 'sigmoid' or type == 'tanh':
            # Assume size: (N, ROWS, COLS) or (ROWS, COLS)
            n_row, n_col = size[-2:]
            value = self.rng_numpy.uniform(
                low=-np.sqrt(6. / (n_row + n_col)),
                high=np.sqrt(6. / (n_row + n_col)),
                size=size)
            if type == 'sigmoid':
                value *= 4.0
        elif type == 'gaussian001':
            value = self.rng_numpy.normal(loc=0, scale=0.01,
                                          size=size)
        elif type == 'uniform':
            n_row = size[-2]
            value = self.rng_numpy.uniform(
                low=-1. / np.sqrt(n_row),
                high=1. / np.sqrt(n_row), size=size)
        else:
            raise Exception('Unknown weight init: ' + type)
        return np.asarray(value, dtype=floatX)

    def get_nade_k_mean_field(self, x, input_mask, k, use_noise=False):
        P = []

        if self.model.train.use_dropout and use_noise:
            mask_size = (k,
                         self.minibatch_size,
                         self.model.n_hidden1 * self.model.n_layers)
            drop_mask = T.cast(self.rng_theano.binomial(p=0.5, size=mask_size),
                               'float32')

        def drop(u, k_, l_):
            if not self.model.train.use_dropout:
                return u
            if use_noise:
                dmask = drop_mask[k_, :, l_ * self.model.n_hidden1:(l_ + 1) * self.model.n_hidden1]
                return u * dmask
            else:
                return constant(1./2.) * u

        assert not self.model.init_mean_field or not self.model.init_mean_field_beta
        if self.model.init_mean_field:
            v = x * input_mask + self.marginal * (1-input_mask)
            #NOTE: The input_mask used in the code and 
            #   that used in the paper have opposite connotations. 
            #   Compare the two formulations to see the difference. 
        elif self.model.init_mean_field_beta:
            #TODO: If we, in the future, choose to use mean field 
            #       with beta training, then we have to make sure
            #       that we understand the mini-section that follows 
            #       immediately. 
            d = T.cast(T.sum(input_mask, axis=-1), 'int32')
            beta = self.betas[d]
            if beta.ndim == 1:
                beta = beta.dimshuffle(0, 'x')
            v = beta * (x - self.marginal) * input_mask + self.marginal
        else:
            v = x * input_mask

        mask = input_mask if self.model.use_mask else T.zeros_like(input_mask)
        #NOTE: As Antti explained to us, you could, if you choose to, 
        #       give the mask as an explicit input to a unit. 
        #      That's what the line above does
        center = lambda x: (x - self.marginal) if self.model.center_v else x
        #Centering implies 'mean' centering - by definition of 'marginal' earlier
        #       in the code. 
        
        #TODO: Try and think about what effect centering the mean and
        #       specifiying the mask as an explicit input has on the 
        #       overall result or on the way optimization is performed. 
        

        for i in range(k):
            if i == 0:
                h = self.activation(T.dot(center(v), self.W1)
                                    + T.dot(mask, self.Wflags)
                                    + self.b1)
            else:
                h = self.activation(T.dot(center(v), self.W1) + T.dot(center(h2),self.V2)+ self.b1)
                #The only difference on every iteration but the first one
                #    is that we don't specify the mask as an explicit input
                #   (assuming that use_mask is set to TRUE). if it is set to
                #   false, it doesn't make a difference. 
            h = drop(h, i, 0)
                #drop only does something if use_droptout and/or use_noise
                #   are set to true. If not, it just returns 'h', unchanged. 

            #TODO: 
            # Figure out how to Modify the 'if' condition below if n_layers>2. 
            if self.model.train.beta_train.active and i == k - 1:
                d = T.cast(T.sum(input_mask, axis=-1), 'int32')
                beta = self.final_betas[d]
                if beta.ndim == 1:
                    beta = beta.dimshuffle(0, 'x')
                p_x_is_one = T.nnet.sigmoid(beta * T.dot(h, self.V.T) + self.c)
            else:
                p_x_is_one = T.nnet.sigmoid(T.dot(h, self.V.T) + self.c)
                # The matrix 'V' is transposed and then multiplied
                #   because, the way V was initialized causes it to 
                #   be incompatible (in shape) with h. 

            if self.model.n_layers == 2:
                # h2 is the output of the second hidden layer of architecture 3
                h2 = self.activation(T.dot(h, self.W2) + self.b2)
                h2 = drop(h2, i, 1)

            # to stabilize the computation
            p_x_is_one = p_x_is_one * constant(0.9999) + constant(0.0001 * 0.5)
            #TODO: Figure out what the line above actually does and why it's needed. 

            # v for the next iteration
            v = x * input_mask + p_x_is_one * (1-input_mask)
            # The values the are intially masked are left unchanged, 
            #   but the values that are missing (and to be predicted) 
            #   are set to the prediction that we've just computed. 

            P.append(p_x_is_one)
        return P

    def estimate_ll(self, data, mb_size):
        if self.ll_est_fn is None:
            self.ll_est_fn = self.compile_ll_est(self.k)
        fn = self.ll_est_fn

        # for testing compute LL
        batches = self.split_into_batches(data, mb_size)

        orderings = [self.rng_numpy.permutation(np.arange(self.model.n_in))
                     for i in range(self.n_orderings)]
        orderings = np.asarray(orderings).astype('int32')

        LLs_all = []
        for k in range(self.n_orderings):
            ordering = orderings[k]
            LLs = []
            for i, batch in enumerate(batches):
                if batch.ndim == 1:
                    batch = batch.reshape(-1, 1)
                if self.verbose:
                    inlineprint('Computing LL', i, '/', len(batches))
                test_LL = fn(batch.T, ordering)
                        #fn = self.ll_est_fn from a couple of lines above
                LLs.append(test_LL)
            LLs = np.concatenate(LLs)
            LLs_all.append(np.mean(LLs))
            print 'this order ', np.mean(LLs), 'average ', np.mean(LLs_all)
        mean_over_orderings = np.mean(LLs_all)
        return mean_over_orderings

    def estimate_ensemble_ll_after_training(self, k, data, n_orderings, mb_size):
        fn = self.compile_ll_est_ensemble(k, n_orderings)
        return self.estimate_ensemble_ll(data, n_orderings, mb_size, fn)

    def estimate_ensemble_ll(self, data, n_orderings, mb_size, fn):
        orderings = [self.rng_numpy.permutation(np.arange(self.model.n_in))
                     for i in range(n_orderings)]
        orderings = np.asarray(orderings).astype('int32')
        lls = []
        to_save = []
        # the table with (orderings, example)
        lls_orderings = []
        batches = self.split_into_batches(data, mb_size)
        to_save = []
        t0 = time.time()
        for i, d in enumerate(batches):
            inlineprint('Computing LL minibatch', i+1, '/', len(batches))
            ll, ll_orderings = fn(d, orderings)
            lls.append(np.mean(ll))
            lls_orderings.append(ll_orderings)
            if (i+1) % 10 == 0:
                use_time = time.time() - t0
                avg_ll = np.mean(lls)
                print 'time %f  mean ll so far %f' % (use_time, avg_ll)

                to_save.append([use_time, avg_ll])
                file = self.save_path + 'MoE_LL_tst_%d_orderings.txt' % n_orderings
                np.savetxt(file, np.asarray(to_save))
                file = self.save_path + 'MoE_LL_tst_%d_orderings_table.npy' % n_orderings
                np.save(file, np.asarray(lls_orderings).T)
        print 'Mean LL', np.mean(lls)
        return np.mean(lls)

    def inpainting(self, epoch, k):
        def compile_inpainting_fn(k):
            x = T.fmatrix('inputs')
            x.tag.test_value = self.input_test_value()

            m = T.fmatrix('masks')
            m.tag.test_value = self.input_test_value()
            input_mask = m
            output_mask = constant(1) - input_mask
            P = self.get_nade_k_mean_field(x, input_mask, k)
            P = T.stacklists(P)
            samples = self.rng_theano.binomial(n=1,p=P,size=P.shape, dtype=floatX)
            samples = samples * output_mask
            fn = theano.function(inputs=[x, m], outputs=samples,name='inpainting_fn')
            return fn
        if not self.inpainting_fn:
            self.inpainting_fn = compile_inpainting_fn(k)
        # generate a square
        input_mask = np.ones((28,28),dtype=floatX)
        input_mask[10:20, 10:20] = np.float32(0)
        input_mask = input_mask.flatten()
        B = 10
        # inpainting how many time each
        N = 10
        xs = self.tstset[:B]
        all_paints = []
        input_mask = input_mask[np.newaxis, :]
        for x in xs:
            x_paints = []
            x = x[np.newaxis, :]
            x_paints.append(x * input_mask)
            for n in range(N):
                # inpaint many times for one x
                x_mis = self.inpainting_fn(x, input_mask)
                inpainted = x_mis + x * input_mask
                a, b, c = inpainted.shape
                inpainted = inpainted.reshape((a * b, c))
                x_paints.append(inpainted)
            all_paints.append(np.concatenate(x_paints, axis=0))
        all_paints = np.concatenate(all_paints, axis=0)
        if DO_IMAGES:
            img = image_tiler.visualize_mnist(data=all_paints, how_many=all_paints.shape[0])
        save_path = self.save_path + 'inpainting_e%d.png' % epoch
        img.save(save_path)

    def train(self):
        self.compile(self.model.cost_from_last)
        # set visible bias, critical
        print 'Main training for', self.n_epochs - 0, 'epochs'
        self.train_with_sgd(self.trnset, epoch=0, epoch_end=self.n_epochs)

        if self.fine_tune_activate:
            # reset the learning rate
            print 'Reset the learning rate for fine-tuning'
            self.learning_rate.set_value(np.float32(self.lr))
            self.lr_decrease = self.lr / self.fine_tune_n_epochs

            self.compile(True)
            #The signature for compile is compile(cost_from_last)
            #   where cost_from_last is a bool

            print 'Start fine tune training'
            self.train_with_sgd(self.trnset, self.n_epochs,
                                self.n_epochs + self.fine_tune_n_epochs)
        else:
            print 'Skipping fine tuning'

        if self.model.train.beta_train.active:
            print 'Start final beta training with validation data'
            self.params = [self.final_betas]
            self.compile(cost_from_last=True)
            self.valid_freq = 1000
            epoch = self.state.epoch if 'epoch' in self.state else 0
            until = self.model.train.beta_train.n_epochs + epoch
            print 'vali ll before training', self.estimate_ll(self.valset, mb_size=5000)
            print 'test ll before training', self.estimate_ll(self.tstset, mb_size=5000)
            self.train_with_sgd(self.valset, epoch, until)
            print 'test ll after training', self.estimate_ll(self.tstset, mb_size=5000)
            self.save_params('after-train-beta-e' + str(until))
        else:
            print 'Skipping beta after-training'

    def train_with_sgd(self, trainset, epoch, epoch_end):
        print 'Train %s with SGD' % self.model.name

        batches = trainset.shape[0] // self.minibatch_size
        minibatches = np.array_split(trainset, batches)
        while (epoch < epoch_end):
            costs_epoch = []
            costs_by_step_epoch = []
            for k, data in enumerate(minibatches):
                if self.verbose:
                    inlineprint('Training minibatch', k, 'of', len(minibatches))
                mask = self.generate_masks(data.shape)
                #NOTE: 
                #   Note that a different mask is generated for each
                #   example for each epoch. 
                
                # len(results)==2
                cost, costs_by_step = self.train_fn(data, mask)
                costs_epoch.append(cost)
                costs_by_step_epoch.append(costs_by_step.flatten())

            cost_epoch_avg = np.mean(costs_epoch)
            cost_by_step_avg = np.asarray(costs_by_step_epoch).mean(axis=0)

            self.costs_steps.append([epoch]+cost_by_step_avg.tolist())
            self.costs.append([epoch, cost_epoch_avg])
            print '\rTraining %d/%d epochs, cost %.2f, costs by step %s' % (
                epoch, epoch_end, cost_epoch_avg, np.round(cost_by_step_avg, 2))
            if epoch != 0 and (epoch+1) % self.valid_freq == 0:
                np.savetxt(self.save_path+'epoch_costs_by_step.txt',
                           self.costs_steps)
                np.savetxt(self.save_path+'epoch_costs.txt', self.costs)
                #self.sample_nade_v0(epoch)
                self.make_plots(self.costs, epoch)
                #self.visualize_filters(epoch)
                self.LL(epoch)
                #self.inpainting(epoch, self.k)
                #self.save_params('e' + str(epoch))
            epoch += 1
            self.state.epoch = epoch
        print

    def sample(self, n, epoch, collect_mean=False, k=None):
        if k is None:
            k = self.k
        if self.sampling_fn is None:
            sampling_fn = self.compile_sampling_fn(k, collect_mean=False)
        else:
            sampling_fn = self.sampling_fn

        samples = self.generate_samples(n, sampling_fn)

        file = 'means' if collect_mean else 'samples'
        file += '_single_ordering_k%d_e%d' % (k, epoch)
        path = os.path.join(self.save_path, file)
        np.save(path + ".npy", samples)

        if DO_IMAGES:
            image_tiler.visualize_mnist(data=samples,
                                        save_path=path + '.png',
                                        how_many=samples.shape[0])
        print

    def generate_samples(self, n, sampling_fn):
        # n: how many samples
        ordering = np.asarray(range(self.model.n_in)).astype('int32')
        self.rng_numpy.shuffle(ordering)
        samples = []
        for i in range(n):
            if self.verbose:
                inlineprint('Sampling', i+1, 'of', n)
            sample = sampling_fn(ordering)
            samples.append(sample)
        # (n,D)
        return np.asarray(samples)

    def make_plots(self, costs, epoch):
        #TODO: Remove line below
        pass
        #try:
        #    import matplotlib.pyplot as plt
        #except ImportError:
        #    return
        #costs = np.asarray(costs)
        #plt.plot(costs[:, 0], costs[:, 1])
        #plt.savefig(self.save_path+'costs_e%d.png' % epoch)

        #if self.model.init_mean_field_beta:
        #    betas = self.betas.get_value()
        #    plt.plot(betas)
        #    plt.savefig(self.save_path+'betas_e%d.png' % epoch)

    def LL(self, epoch):
        print 'estimate LL on validset'
        valid_LL = self.estimate_ll(self.valset, 5000)

        self.LL_valid.append([epoch, valid_LL])

        t = np.asarray(self.LL_valid)
        best_idx = np.argmax(t[:, 1])
        best_epoch = t[best_idx, 0]
        best_valid = t[best_idx, 1]
        self.state['best_validset_LL'] = best_valid
        self.state['best_epoch'] = best_epoch
        print 'Current valid LL %.3f, best valid LL %.3f at epoch %d' % (
            valid_LL, best_valid, best_epoch)

        if best_idx + 1 == len(self.LL_valid):
            self.save_params("best")

        np.savetxt(self.save_path+'valid_LL.txt', self.LL_valid)
        return valid_LL

    def visualize_filters(self, epoch):
        print 'saving filters'
        to_do = [self.W1.get_value(), self.V.get_value()]
        names = ['W1', 'V']
        for param, name in zip(to_do, names):
            filters = image_tiler.visualize_first_layer_weights(param, [28,28])
            name = self.save_path + 'filters_e%d_%s.png'%(epoch,name)
            filters.save(name)

    def sample_after_train(self, k, count, fn, collect_mean=False):
        # pick a random ordering and then sample
        sampling_fn = self.compile_sampling_fn(k, collect_mean)
        ordering = np.asarray(range(self.model.n_in)).astype('int32')
        self.rng_numpy.shuffle(ordering)
        samples = []
        for i in range(count):
            inlineprint('Sampling', i+1, count)
            sample = sampling_fn(ordering)
            samples.append(sample)
        samples = np.asarray(samples)

    def save_params(self, name):
        path = self.save_path + 'model_params_%s.pkl' % name
        print 'Saving model params to', path
        params = [param.get_value() for param in self.params]
        self.best_params = params
        with open(path, 'wb') as f:
            params = cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        self.config.save('state-%s' % name)

    def load_params(self, params_path):
        print 'Loading learned parameters from %s' % params_path
        with open(params_path, 'rb') as f:
            params = cPickle.load(f)
        print 'Found', len(params), 'params'
        self.set_params(params)

    def set_params(self, params):
        assert len(self.params) == len(params)
        for param_new, param_old in zip(params, self.params):
            print 'Setting', param_old.name
            assert param_new.shape == param_old.get_value().shape
            param_old.set_value(param_new)

    def show_mean_field_inference(self):
        x = T.fmatrix('inputs')
        x.tag.test_value = self.input_test_value()
        m = T.fmatrix('masks')
        m.tag.test_value = self.input_test_value()

        data = self.tstset[:20]
        means = self.get_nade_k_mean_field(x, m, 10)
        fn = theano.function([x, m], means)
        mask = self.generate_masks(data.shape)
        outputs = fn(data, mask)
        init = data * mask + self.marginal * (1-mask)
        masked = data * mask + 0.5 * (1-mask)
        # (7,5,784)
        to_visualize = [data, masked, init] + outputs
        # (5, 7, 784)
        t = np.transpose(to_visualize, (1,0,2))
        a, b, c = t.shape
        t = t.reshape((a*b, c))
        image_tiler.visualize_mnist(data=t,
                                    save_path=self.save_path + 'mean_field_test.png',
                                    how_many=t.shape[0],tile_shape=(20,13))


def train_from_scratch(config, data):
    model = NADEk(config, data)
    model.init_params()
    model.train()
    model.set_params(model.best_params)
    model.estimate_ll(data=data[1], mb_size=5000)
    model.estimate_ll(data=data[2], mb_size=5000)
    model.estimate_ensemble_ll_after_training(k=config.model.train.k,
                                              data=data[2],
                                              n_orderings=64,
                                              mb_size=1000)


def evaluate_trained(config, data, params_file):
    model = NADEk(config, data)
    model.init_params()
    model.load_params(params_file)
    model.compile(model.cost_from_last)
    model.estimate_ll(data[1], 5000) #Estimate the LL on the validation data
    model.estimate_ll(data[2], 5000) #Estimate the LL on the test data
    model.estimate_ensemble_ll_after_training(k=config.model.train.k,
                                              data=data[2],
                                              n_orderings=128,
                                              mb_size=1000)


def continue_train(config, data, params_file):
    config.save()
    model = NADEk(config, data)
    model.init_params()
    model.load_params(params_file)
    model.train()
