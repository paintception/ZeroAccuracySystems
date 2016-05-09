import tensorflow as tf
sess = tf.InteractiveSession()
#x = tf.constant([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])
x = tf.constant([[1,2],[3,4],[5,6],[7,8]])

print("X")
print(x)
print(sess.run(x))

print("T")
#t = tf.transpose(x,[2,1,0])
#t = tf.reshape(x,[2,-1,2])
t = tf.split(0,2,x)
for ts in t:
    print(ts)
    print(sess.run(ts))