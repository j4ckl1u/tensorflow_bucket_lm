import tensorflow as tf
import ZoneOutLSTM
import Config
import  numpy as np

class LM_Model:
    def __init__(self):
        scope = "LM_Model"
        with tf.variable_scope(scope):
            self.Emb = tf.get_variable("Embedding", shape=[Config.VocabSize, Config.EmbeddingSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wt = tf.get_variable("pWeight", shape=[Config.HiddenSize, Config.VocabSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wtb = tf.get_variable("pBias", shape=[Config.VocabSize],
                                       initializer=tf.constant_initializer(0.0))
            self.firstHidden = tf.constant(0.0, shape=[Config.BatchSize, Config.HiddenSize])
            self.Decoder = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.HiddenSize, scope=scope)

        self.Parameters = [self.Emb, self.Wt, self.Wtb]
        self.Parameters.extend(self.Decoder.Parameters)

    def create_network(self, inputTrg, maskTrg, length, optimizer):
        network_hidden_trg = {}
        network_mem_trg = {}
        tce = 0
        for i in range(0, length, 1):
            if i == 0:
                network_hidden_trg[i] = self.firstHidden
                network_mem_trg[i] = network_hidden_trg[i]
            else:
                embed = tf.nn.embedding_lookup(self.Emb, inputTrg[i-1])
                (network_hidden_trg[i], network_mem_trg[i]) = self.Decoder.createNetwork(embed, network_hidden_trg[i - 1],
                                                                                       network_mem_trg[i-1])
            logits_out = tf.matmul(network_hidden_trg[i], self.Wt) + self.Wtb
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=inputTrg[i])
            tce += tf.reduce_sum(ce*maskTrg[i])
        totalCount = tf.reduce_sum(maskTrg)
        tce = tce/totalCount
        min_loss = optimizer.minimize(tce)
        return min_loss, tce
