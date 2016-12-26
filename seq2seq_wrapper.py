import tensorflow as tf
import numpy as np


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, xvocab_size, yvocab_size, batch_size, emb_dim, num_layers):
        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self

        # placeholders
        #  encoder inputs : list of indices of length xseq_len
        self.enc_ip = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int64, 
                        name='ei_{}'.format(t) for t in range(xseq_len) ]

        #  labels that represent the real outputs
        self.labels = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int64, 
                        name='ei_{}'.format(t) for t in range(yseq_len) ]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)
        # define the basic cell
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
        # stack cells together : n layered model
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


        # for parameter sharing between training model
        #  and testing model
        with tf.variable_scope('decoder') as scope:
            # build the seq2seq model 
            #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
            self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                xvocab_size, yvocab_size, emb_dim)
            # share parameters
            scope.reuse_variables()
            # testing model, where output of previous timestep is fed as input 
            #  to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                feed_previous=True)

        # now, for training,
        #  build loss function

        # weighted loss
        #  TODO : add parameter hint
        loss_weights = [ tf.ones_like(label, dtype=tf.int32) for label in self.labels ]
        self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
        # train op to minimize the loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)


    '''
        Training and Evaluation

    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([loss, decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

        
            










