"""
Template from the magnificient ML lectures of Hugo Larochelle.
Caner Mercan, 2017.

"""
#from mlpython.learners.generic import Learner
import numpy as np
import pdb # for debugging

#class LinearChainCRF(Learner):
class LinearChainCRF():
    """
    Linear chain conditional random field. The contex window size
    has a radius of 1.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constante for the learning rate.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``:    Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 L2=0.001,
                 L1=0,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.L2=L2
        self.L1=L1
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations
        and initializes the parameters of the CRF to 0 (DONE)
        """

#        TODO: 
#        Check the possible values of target 
#        if the target values do not range from 0 to n_classes-1, 
#        aka transformation on the target values so that they follow this rule.
                
        self.n_classes = n_classes
        self.input_size = input_size

        # Can't allocate space for the alpha/beta tables of
        # belief propagation (forward-backward), since their size
        # depends on the input sequence size, which will change from
        # one example to another.

        self.alpha = np.zeros((0,0)) # []
        self.beta = np.zeros((0,0)) # []
        
        ###########################################
        # Allocate space for the linear chain CRF #
        ###########################################
        # - self.weights[0] are the connections with the image at the current position
        # - self.weights[-1] are the connections with the image on the left of the current position
        # - self.weights[1] are the connections with the image on the right of the current position
        self.weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        # - self.bias is the bias vector of the output at the current position
        self.bias = np.zeros((self.n_classes))

        # - self.lateral_weights are the linear chain connections between target at adjacent positions
        self.lateral_weights = np.zeros((self.n_classes,self.n_classes)) # V matrix
        
        self.grad_weights = [np.zeros((self.input_size,self.n_classes)),
                             np.zeros((self.input_size,self.n_classes)),
                             np.zeros((self.input_size,self.n_classes))]
        self.grad_bias = np.zeros((self.n_classes))
        self.grad_lateral_weights = np.zeros((self.n_classes,self.n_classes))
                    
        #########################
        # Initialize parameters #
        #########################

        # Since the CRF log factors are linear in the parameters,
        # the optimization is convex and there's no need to use a random
        # initialization.

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.n_classes)
        self.epoch = 0
        
    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """

        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            self.initialize(input_size,n_classes)
            
        for it in range(self.epoch,self.n_epochs):
            for input,target in trainset:
                self.fprop(input,target)
                self.bprop(input,target)
                self.update()
        self.epoch = self.n_epochs
        
    def fprop(self,input,target):
        """
        Forward propagation: 
        - computes the value of the unary log factors for the target given the input (the field
          self.target_unary_log_factors should be assigned accordingly)
        - computes the alpha and beta tables using the belief propagation (forward-backward) 
          algorithm for linear chain CRF (the field ``self.alpha`` and ``self.beta`` 
          should be allocated and filled accordingly)
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 2D array where the number of
        rows is the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """
        # (your code should call belief_propagation and training_loss)         
        n_inputs = len(target)
        self.target_unary_log_factors = np.zeros((self.n_classes, n_inputs))
        for k in range(n_inputs):
            log_factor = np.zeros(self.n_classes)
            #log_factor += np.dot(self.weights[0].T, input[k-1]) if k > 0 else 0
            #log_factor += np.dot(self.weights[1].T, input[k]) 
            #log_factor += np.dot(self.weights[2].T, input[k+1]) if k < n_inputs-1 else 0
            log_factor += np.dot(self.weights[-1].T, input[k-1]) if k > 0 else 0
            log_factor += np.dot(self.weights[0].T, input[k]) 
            log_factor += np.dot(self.weights[1].T, input[k+1]) if k < n_inputs-1 else 0
            log_factor += self.bias
            self.target_unary_log_factors[:,k] = log_factor

        # compute alpha/beta tables                                              
        self.alpha, self.beta = self.belief_propagation(input) 
        # compute the training loss on the current input                  
        train_loss = self.training_loss(target, self.target_unary_log_factors, self.alpha, self.beta)           

        return train_loss
        

    def belief_propagation(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """

        # alpha computation
        n_inputs = len(input)        
        alpha    = np.zeros([self.n_classes, n_inputs])
        for k in range(n_inputs):    
            aux = 0             
            aux += self.lateral_weights if k < n_inputs-1 else 0
            aux += np.tile(alpha[:, k-1], (self.n_classes, 1)).T if k > 0 else 0    # columnwise operations are so HARD!      
            aux += np.tile(self.target_unary_log_factors[:,k], (self.n_classes, 1)).T # columnwise operations are so HARD!
            alpha[:,k] = self.logsumexp(aux, axs=0)
        # verification of alpha computation (as well as beta computation since the last column of alpha and the first column of beta should give the same outputs: Z(X))
#        alpha2   = np.zeros([self.n_classes, n_inputs])
#        for k in range(n_inputs):
#            for c_yk1 in range(self.n_classes): # yk+1
#                aux = []
#                for c_yk in range(self.n_classes): # yk
#                    aux2 = 0
#                    aux2 += self.target_unary_log_factors[c_yk,k]
#                    aux2 += self.lateral_weights[c_yk,c_yk1] if k < n_inputs-1 else 0
#                    aux2 += alpha[c_yk,k-1] if k > 0 else 0
#                    aux.append(aux2)
#                alpha2[c_yk1,k] = self.logsumexp(aux)
                       
        beta = np.zeros([self.n_classes, n_inputs])
        for k in range(n_inputs-1, -1, -1):
            bux = 0
            bux += self.lateral_weights     if k > 0 else 0
            bux += beta[:, k+1]             if k < n_inputs-1 else 0
            bux += self.target_unary_log_factors[:,k] 
            beta[:,k] += self.logsumexp(bux.T, axs=0) # cant do columnwise subtraction, hence take the transpose, then send!
            

        #print('Z(X) from alpha table: ', alpha[0,-1])
        #print('Z(X) from beta table: ', beta[0,0])
        
        #assert(np.abs(alpha[0,-1] - beta[0,0]) < 1e-10) # results from alpha and beta tables should be the same!
        if np.abs(alpha[0,-1] - beta[0,0]) > 1e-10: # results from alpha and beta tables should be the same!
            print('Z(X) from alpha table: ', alpha[0,-1])
            print('Z(X) from beta table: ', beta[0,0])
            pdb.set_trace()
        
        return np.copy(alpha), np.copy(beta)

#        raise NotImplementedError()
    

    def training_loss(self,target,target_unary_log_factors,alpha,beta):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given the true target, the unary log factors of the target space and alpha/beta tables
        """

        unary_fact 	= np.sum(target_unary_log_factors[target, range(0, len(target))])  # sum_k=1^K a_u(y_k)  p.s. code starts from zero, not one.
        pairwise_fact 	= np.sum(self.lateral_weights[target[:-1], target[1:]]) # sum_k=1^K-1 a_p(y_k,y_{k+1})  p.s. code starts from zero, not one.
        log_fact 		= unary_fact + pairwise_fact # log(exp(unary_fact+pairwise_fact))
        log_part_fun 	= alpha[0,-1] # == beta[0,0] ## logZ(X) ##
        neg_loglik 	= -log_fact + log_part_fun # -log p(y|X) where y consists of the labels of the current sequence (target)
        
        reg_term_L2 = self.L2 * np.sum([np.linalg.norm(W, ord='fro')**2 for W in self.weights]) # computing frobenius norm from the linalg library
#       reg_term_L2     = self.L2 * np.sum([np.trace(np.dot(W, W.T)) for W in self.weights]) # same as above frobnorm(A) = sqrt(trace(A*A.T))
        reg_term_L1 = self.L1 * np.sum([np.sum(np.abs(W)) for W in self.weights])
   
#        loss = neg_loglik + reg_term_L2
        loss = neg_loglik + reg_term_L2 + reg_term_L1
        return loss
  
#        raise NotImplementedError()

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the CRF gradients of the weights, lateral weights and bias 
          in self.grad_weights, self.grad_lateral_weights and self.grad_bias
        - returns nothing
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """
        
        #pdb.set_trace()

        # computation of p(y_k|X)
        p_marg = self.marg_p_yk(target)        
        # one hot vector: e(y_k)
        e_onehot = np.zeros((len(target), self.n_classes))
        e_onehot[range(len(target)), target] = 1

        # bias (b) gradient
        self.grad_bias = np.sum( -(e_onehot-p_marg) , axis=0 )
        # weight (W) gradients; W[0], W[1], W[2]
        self.grad_weights = [np.zeros((self.input_size,self.n_classes)),
                             np.zeros((self.input_size,self.n_classes)),
                             np.zeros((self.input_size,self.n_classes))]
        for k in range(len(target)):
            self.grad_weights[-1] += np.outer( input[k-1], -(e_onehot[k]-p_marg[k])) if k > 0 else 0
            self.grad_weights[0] += np.outer( input[k],   -(e_onehot[k]-p_marg[k]))
            self.grad_weights[1] += np.outer( input[k+1], -(e_onehot[k]-p_marg[k])) if k < len(target)-1 else 0
        # regularization gradient for weight gradients.
        for i in range(len(self.grad_weights)):
            self.grad_weights[i] += self.L2 * (2 * self.weights[i])         # L2 regularization gradient
            self.grad_weights[i] += self.L1 * (np.sign(self.weights[i]))    # L1 regularization gradient
		# lateral weight (V) gradients	
        p_marg_pair = self.marg_p_yk_yk1(target)
        aux = np.zeros(self.grad_lateral_weights.shape)
        for k in range(len(target)-1):
            e_onemat = np.zeros((self.n_classes, self.n_classes))
            e_onemat[target[k], target[k+1]] = 1
            aux += -(e_onemat - p_marg_pair[k])
        self.grad_lateral_weights = aux


    def marg_p_yk(self, target):
        """
        For each input in the sequence;
        Computes the single marginal probability of an input 
        for every possible value(label) it can take. 
        returns a matrix with size; len(target) x n_classes.
        
        """	
        p_yk = []
        K = len(target)
        for k in range(K):
            au_yk = self.target_unary_log_factors[:,k]
            alpha = self.alpha[:, k-1] if k > 0 else 0
            beta  = self.beta[:, k+1]  if k < K-1 else 0
            aux   = np.exp(au_yk + alpha + beta)
            p_yk.append(aux/np.sum(aux))
            #p_yk = np.array(marg)

        # verification of p_yk in multiple for loops -less efficient implementation.
#        p_yk = []
#        for k in range(K):
#            aux_c = []
#            for c in range(self.n_classes):
#                aux = 0
#                aux += self.target_unary_log_factors[c,k]
#                aux += self.alpha[c,k-1] if k > 0 else 0
#                aux += self.beta[c,k+1] if k < K-1 else 0
#                aux_c.append(np.exp(aux))
#            p_yk.append([aux_c / np.sum(aux_c)])
        
        return p_yk		
	
    def marg_p_yk_yk1(self, target):
        """
        For each input in the sequence;
        Computes the marginals of a pair of target variables 
        (y_k and y_k+1) for every possible values [0, n_classes-1] they can take. 
        returns len(target)-1 matrices with sizes; n_classes x n_classes.
        """	

        p_yk_yk1 = []
        K = len(target)
        for k in range(K-1):
            a_p 		= self.lateral_weights
            au_yk 		= np.tile(self.target_unary_log_factors[:,k][np.newaxis].T, [1, self.n_classes]) # columnwise repmat requires transpose for which array is now a matrix.
            au_yk1 		= np.tile(self.target_unary_log_factors[:,k+1], [self.n_classes, 1]) # rowwise repmat works default way.
            log_alpha 	= np.tile(self.alpha[:, k-1][np.newaxis].T, [1, self.n_classes]) if k > 0 else 0
            log_beta 	= np.tile(self.beta[:, k+2], [self.n_classes, 1])  if k < K-2 else 0 	
            aux = 0
            aux = np.exp(au_yk + au_yk1 + a_p + log_alpha + log_beta)
            aux = aux / np.sum(aux)
            p_yk_yk1.append(aux)
        return p_yk_yk1
    
    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the CRF parameters self.weights,
          self.lateral_weights and self.bias, using the gradients in 
          self.grad_weights, self.grad_lateral_weights and self.grad_bias
        """    

        #pdb.set_trace()
        self.bias -= self.lr * self.grad_bias
        self.lateral_weights -= self.lr * self.grad_lateral_weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.grad_weights[i]
           
    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a list of size ``len(dataset)``, containing
          a Numpy 1D array that gives the class prediction for each position
          in the sequence, for each input sequence in ``dataset``
        Argument ``dataset`` is an MLProblem object.
        """
 
        outputs = []

        for input,target in dataset:
            self.fprop(input, target)
            p_yk = self.marg_p_yk(target)      
            outputs.append(np.argmax(p_yk, axis=1))
            
        return outputs
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of the
        CRF for ``dataset``:
        - the errors should be a list of size ``len(dataset)``, containing
          a pair ``(classif_errors,nll)`` for each examples in ``dataset``, where 
            - ``classif_errors`` is a Numpy 1D array that gives the class prediction error 
              (0/1) at each position in the sequence
            - ``nll`` is a positive float giving the regularized negative log-likelihood of the target given
              the input sequence
        Argument ``dataset`` is an MLProblem object.
        """
        
        errors  = []
        outputs = self.use(dataset)
        
        cntr = 0
        for input, target in dataset:
            nll = self.fprop(input, target)
            cl_err = np.argmax(outputs[cntr]) != np.argmax(target)
            errors.append((cl_err, nll))
            cntr += 1
        
        return outputs, errors
 
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """
        
        print('WARNING: calling verify_gradients reinitializes the learner')
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.initialize(10,3)
        example = (rng.rand(4,10),np.array([0,1,1,2]))
        input,target = example
        epsilon = 1e-6
        self.lr = 0.1
        self.decrease_constant = 0

        self.weights = [0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes)]
        self.bias = 0.01*rng.rand(self.n_classes)
        self.lateral_weights = 0.01*rng.rand(self.n_classes,self.n_classes)
        
        self.fprop(input,target)
        self.bprop(input,target) # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)
  
        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i,j] += epsilon
                    a = self.fprop(input,target)
                    self.weights[h][i,j] -= epsilon
                    
                    self.weights[h][i,j] -= epsilon
                    b = self.fprop(input,target)
                    self.weights[h][i,j] += epsilon
                    
                    #pdb.set_trace()
                    
                    emp_grad_weights[h][i,j] = (a-b)/(2.*epsilon)        

        print('grad_weights[-1] diff.:',np.sum(np.abs(self.grad_weights[-1].ravel()-emp_grad_weights[-1].ravel()))/self.weights[-1].ravel().shape[0])
        print('grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0])
        print('grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0])
  
        emp_grad_lateral_weights = copy.deepcopy(self.lateral_weights)
  
        for i in range(self.lateral_weights.shape[0]):
            for j in range(self.lateral_weights.shape[1]):
                self.lateral_weights[i,j] += epsilon
                a = self.fprop(input,target)
                self.lateral_weights[i,j] -= epsilon

                self.lateral_weights[i,j] -= epsilon
                b = self.fprop(input,target)
                self.lateral_weights[i,j] += epsilon
                
                emp_grad_lateral_weights[i,j] = (a-b)/(2.*epsilon)


        print('grad_lateral_weights diff.:',np.sum(np.abs(self.grad_lateral_weights.ravel()-emp_grad_lateral_weights.ravel()))/self.lateral_weights.ravel().shape[0])

        emp_grad_bias = copy.deepcopy(self.bias)
        for i in range(self.bias.shape[0]):
            self.bias[i] += epsilon
            a = self.fprop(input,target)
            self.bias[i] -= epsilon
            
            self.bias[i] -= epsilon
            b = self.fprop(input,target)
            self.bias[i] += epsilon
            
            emp_grad_bias[i] = (a-b)/(2.*epsilon)

        print('grad_bias diff.:',np.sum(np.abs(self.grad_bias.ravel()-emp_grad_bias.ravel()))/self.bias.ravel().shape[0])
        
        
    
    def logsumexp(self, vals, axs=0):
        """
        Computes numerically stable version of logsumexp operation.
        """
        maxval = np.max(vals, axis=axs)
        lsexp  = maxval + np.log(np.sum(np.exp(vals - maxval), axis=axs))
        return lsexp
        
