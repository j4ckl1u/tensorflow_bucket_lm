from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math
import Config
import RnnlmModel
import Corpus

class RNNLM_Trainer:

    def __init__(self):
        self.model = RnnlmModel.LM_Model()
        self.trainData = Corpus.MonoCorpus(Config.trgVocabF, Config.trainTrgF)
        self.valData = Corpus.MonoCorpus(Config.trgVocabF, Config.valTrgF)
        self.networkBucket = {}

        self.inputTrg = tf.placeholder(tf.int32, shape=[Config.MaxLength, Config.BatchSize],
                                   name='input')
        self.maskTrg = tf.placeholder(tf.float32, shape=[Config.MaxLength, Config.BatchSize], name='inputMask')
        self.optimizer = tf.train.AdamOptimizer()
        self.createBucketNetworks(Config.MaxLength)


    def getNetwork(self, lengthTrg):
        bucketGap = Config.BucketGap
        networkTrgid = int(math.ceil(lengthTrg/float(bucketGap))*bucketGap)
        networkTrgid = networkTrgid if networkTrgid <= Config.MaxLength else Config.MaxLength
        if(networkTrgid not in self.networkBucket):
            print("Creating network (" + str(networkTrgid) + ")", end="\r")
            self.networkBucket[networkTrgid] = self.model.create_network(self.inputTrg, self.maskTrg, networkTrgid, self.optimizer)
            print("Bucket contains networks for ", end="")
            for key in self.networkBucket: print("(" + str(key) + ")", end=" ")
            print()
        return self.networkBucket[networkTrgid]

    def createBucketNetworks(self, maxLength):
        maxBucket = int(math.ceil(maxLength/float(Config.BucketGap))*Config.BucketGap)
        for length in range(Config.BucketGap, maxBucket+1, Config.BucketGap):
            self.getNetwork(length)

    def train(self):
        cePerWordBest = 10000
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(0, 100000, 1):
            print("Traing with batch " + str(i), end="\r")
            trainBatch = self.trainData.getTrainBatch()
            maxTrgLength = max(len(x) for x in trainBatch)
            min_loss, loss = self.getNetwork(maxTrgLength)
            (batch, batchMask) = self.buildInput(trainBatch)
            train_dict = {self.inputTrg: batch, self.maskTrg: batchMask}
            _, cePerWord = sess.run([min_loss, loss], feed_dict=train_dict)
            if(i % 10 == 0):
                print(str(cePerWord/math.log(2.0)))

    def buildInput(self, sentences):
        vocabSize = Config.VocabSize
        maxLength = Config.MaxLength

        batch = np.zeros((maxLength, Config.BatchSize), dtype=np.int32)
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j])):
                    batch[i, j] = sentences[j][i]
                else:
                    batch[i, j] = self.trainData.getEndId()
        batchMask = np.zeros((maxLength, Config.BatchSize), dtype=np.float32)
        for i in range(0, len(sentences), 1):
            sentence = sentences[i]
            lastIndex = len(sentence) if len(sentence) < maxLength else maxLength
            batchMask[0:lastIndex, i] = 1

        return (batch, batchMask)

if __name__ == '__main__':
    rnnLMTrainer = RNNLM_Trainer()
    rnnLMTrainer.createBucketNetworks(Config.MaxLength)
    rnnLMTrainer.train()
