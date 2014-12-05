#!/usr/bin/python
import random
import os
import numpy as np
import math
import time
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=160)
plt.figure(figsize=(16,16))

class rbm(object):
  def __init__(self, numVisible, numHidden, learningRate, verbose=True):
    self.numVisible = numVisible
    self.numHidden = numHidden
    self.learningRate = learningRate

    self.visibleLayer = np.zeros(self.numVisible)
    self.hiddenLayer = np.zeros(self.numHidden)

    self.visibleLayer2 = np.zeros(self.numVisible)
    self.hiddenLayer2 = np.zeros(self.numHidden)

    self.weights = np.random.normal(size=(self.numHidden, self.numVisible), scale=0.01)
    self.visibleBias = np.zeros(self.numVisible)
    self.hiddenBias = np.zeros(self.numHidden)

    self.verbose = verbose

  def sigmoidTransform(self, a):
    return 1.0/(1.0 + np.exp(-1.0*a))

  def sample(self, probA, n):
    x = np.random.random(n)
    return 1*(x < probA)

  def train(self, numIterations, data):
    numDigits = data.shape[0]
    start = time.time()
    for i in xrange(numIterations):
      for j in xrange(numDigits):
        if j % 1000 == 0 and j != 0 and self.verbose:
          print "**current iteration: " + str(i+1) + " digits trained: " + str(j)
        self.visibleLayer = data[j,:]
        self.hiddenLayer = self.sample(self.sigmoidTransform(np.dot(self.weights, self.visibleLayer) + self.hiddenBias), self.numHidden)

        self.visibleLayer2 = self.sample(self.sigmoidTransform(np.dot(self.weights.T, self.hiddenLayer) + self.visibleBias), self.numVisible)
        self.hiddenLayer2 = self.sigmoidTransform(np.dot(self.weights, self.visibleLayer2) + self.hiddenBias)
        
        positiveGradient = np.outer(self.hiddenLayer, self.visibleLayer)
        negativeGradient = np.outer(self.hiddenLayer2, self.visibleLayer2)

        update = self.learningRate*(positiveGradient - negativeGradient)
        # print np.mean(np.abs(update))/np.mean(np.abs(self.weights))
        self.weights += update
        self.visibleBias += self.learningRate*(self.visibleLayer - self.visibleLayer2)
        self.hiddenBias += self.learningRate*(self.hiddenLayer - self.hiddenLayer2)

        # print self.weights
      print "iterations: " + str(i+1) + " time elapsed: " + str(time.time() - start)

  def transform(self, mat):
    if len(mat.shape) == 1:
      return self.sigmoidTransform(np.dot(self.weights, mat) + self.hiddenBias)
    else:
      newMat = np.empty((mat.shape[0], self.numHidden))
      for i in xrange(mat.shape[0]):
        newMat[i,:] = self.sigmoidTransform(np.dot(self.weights, mat[i,:]) + self.hiddenBias)
      return newMat

  def visualizeFilters(self, n=100, dim=28):
    print "visualizing filters"
    plotDim = int(math.ceil(math.sqrt(n)))
    for i in xrange(n):
      image = self.weights[i].reshape((dim,dim))
      plt.subplot(plotDim,plotDim,i+1)
      plt.set_cmap('binary')
      plt.axis('off')
      plt.imshow(image, interpolation='nearest')
    plt.savefig("filters.png", bbox_inches='tight')

  def genSamplesHelper(self, myInput, n=100, chainInterval=1000):
    samples = []
    self.visibleLayer = myInput
    # print input
    for i in xrange(n):
      for j in xrange(chainInterval):
        self.hiddenLayer = self.sample(self.sigmoidTransform(np.dot(self.weights, self.visibleLayer) + self.hiddenBias), self.numHidden)
        # print np.dot(self.weights, self.visibleLayer)
        self.visibleLayer = self.sample(self.sigmoidTransform(np.dot(self.weights.T, self.hiddenLayer) + self.visibleBias), self.numVisible)
        # print self.sigmoidTransform(np.dot(self.weights.T, self.hiddenLayer)).reshape((28,28))
        samples.append(np.copy(self.visibleLayer))
    return samples

  def generateSamples(self, data, d=10, n=9, dim=28):
    print "generating samples"
    # plotDim = int(math.ceil(math.sqrt(n)))
    counter = 1
    for i in xrange(d):
      myInput = data[random.randrange(data.shape[0]),:]
      origImage = myInput.reshape((dim, dim))
      plt.subplot(d,n+1,counter)
      plt.imshow(origImage, interpolation='nearest')
      plt.set_cmap('gray')
      plt.axis('off')
      counter += 1

      samples = self.genSamplesHelper(myInput, n=n)
      for i in xrange(n):
        sampleImage = samples[i].reshape((dim,dim))
        plt.subplot(d,n+1,counter)
        plt.imshow(sampleImage, interpolation='nearest')
        plt.set_cmap('binary')
        plt.axis('off')
        counter += 1
    plt.savefig("samples.png", bbox_inches='tight')

  def exportParams(self, fileName):
    np.savez(fileName, weights=self.weights, hiddenBias=self.hiddenBias, visibleBias=self.visibleBias)
    print "weights successfully exported"

  def importParams(self, fileName):
    npzfile = np.load(fileName)
    self.weights = npzfile["weights"]
    self.hiddenBias = npzfile["hiddenBias"]
    self.visibleBias = npzfile["visibleBias"]
    print "weights successfully imported"

if __name__ == '__main__':
  mnist = fetch_mldata('MNIST original')
  permuteIndex = np.random.permutation(mnist.data.shape[0])
  mnist.data = mnist.data/255.0
  mnist.data = mnist.data[permuteIndex,:]
  mnist.target = mnist.target[permuteIndex]

  myData = np.copy(mnist.data)
  index = myData > 126/255.0
  # print index
  myData[index] = 1
  myData[~index] = 0

  # print myData[0]
  # print myData[0].reshape((28,28))
  # for i in xrange(myData.shape[0]):
  #   print i
  #   index = myData[i] > 126
  #   myData[index] = 1
  #   myData[~index] = 0

  x = rbm(784, 500, 0.005)
  
  try:
    x.importParams("trainedRBM.npz")
  except:
    x.train(15, myData[:60000,:])
    x.exportParams("trainedRBM.npz")

  # x.visualizeFilters()
  # x.generateSamples(myData[60000:,:])
  # from sklearn.svm import SVC
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression()
  d = 60000

  print "without rbm"
  clf.fit(mnist.data[:d,:], mnist.target[:d])
  print "accuracy: ", clf.score(mnist.data[d:,:], mnist.target[d:])

  print "with rbm"
  clf.fit(x.transform(mnist.data[:d,:]), mnist.target[:d])
  print "accuracy: ", clf.score(x.transform(mnist.data[d:,:]), mnist.target[d:])

  # without rbm
  # accuracy:  0.918
  # with rbm
  # accuracy:  0.9693
