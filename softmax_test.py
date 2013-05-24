import theano, numpy
x = theano.shared(numpy.zeros((2,9999), dtype='float32'))
y = theano.shared(numpy.ones((2,9999), dtype='float32'))
s = theano.tensor.nnet.softmax(x + y)
f = theano.function([], s)
f()
