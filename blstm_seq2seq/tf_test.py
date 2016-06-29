import tensorflow as tf

#t1 = tf.Variable([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[100,200,300],[400,500,600]],[[700,800,900],[1000,1100,1200]]]])
#t1 = tf.Variable([[[[1111],[1121],[1131]],[[1211],[1221],[1231]]]])
# Batch size (1x) => Time steps (2x) => Features (3x) => Channels (2x)
t1 = tf.Variable([[[[1111,1112],[1121,1122],[1131,1132]],[[1211,1212],[1221,1222],[1231,1232]]]])
# Batch size (1x) => Time steps (2x) => Features*Channels (6x)
t2 = tf.Variable([[[1111,1112,1121,1122,1131,1132],[1211,1212,1221,1222,1231,1232]]])
t3 = tf.reshape(t1,[-1,2,6])
t4 = tf.reshape(t3,[-1,2,6,1])
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(t1.get_shape())
    print(sess.run(t1))

    print("===")

    print(t2.get_shape())
    print(sess.run(t2))

    print("===")

    print(t3.get_shape())
    print(sess.run(t3))

    print("===")

    print(t4.get_shape())
    print(sess.run(t4))