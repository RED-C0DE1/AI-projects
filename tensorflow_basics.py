import tensorflow as tf
print(tf.__version__)

# 1. Initialization Tensors

x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,4], [4,5,6]])

# generate zeros
x = tf.zeros((3,3))

# one hot encoding
x = tf.eye(3)

# standardise normal dist
x = tf.random.normal((3,3), mean=0, stddev=1)

# uniform dist
x = tf.random.uniform((1,3), minval=0, maxval=1)

# change dtype
x = tf.cast(x, dtype=tf.float64)

# 2. Mathematical Operations

# add
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
z = x + y

# subtract
z = x - y

# div
z = z / y

# mult
z = x * y

# dot product
z = tf.tensordot(x, y, axes=1)

# sqr
z = x ** 5

# matrix mult
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = x @ y
print(z)

# 3. Indexing
x = tf.constant([0,1,1,2,3,1,2,3])
# print everything except first value
print(x[1:])

# print after first index and stop at 3
print(x[1:3])

# take 2 steps when going throgh data
print(x[::2])

# print in reverse
print(x[::-1])

indices = tf.constant([0,3])
x_ind = tf.gather(x, indices)
print(x_ind)

# get certain rows of elements
x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
print(x[0,:])
print((x[0:2]))

# 4. Reshaping
x = tf.range(9)
x = tf.reshape(x, (3,3))

x = tf.transpose(x, perm=[1,0])
