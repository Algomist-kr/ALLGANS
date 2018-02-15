from model.AbstractGANModel import AbstractGANModel
from util.SequenceModel import SequenceModel
from util.ops import *
from util.summary_func import *
from dict_keys.dataset_batch_keys import *
import numpy as np


class Vertex:
    def __init__(self, id_, layer, tensor=None):
        self.id_ = id_
        self.layer = layer
        self.tensor = None
        self.next = []
        self.from_ = []

    def __repr__(self):
        return "%s id:%s, layer:%s,  from=%s, next=%s" % (
            self.__class__.__name__, self.id_, self.layer, self.from_, self.next)

    def add_next(self, id_):
        self.next += [id_]
        self.next.sort()

    def add_from(self, id_):
        self.from_ += [id_]
        self.from_.sort()

    def is_connected(self, other):
        if other.id_ in self.next or other.id_ in self.from_:
            return True
        else:
            return False


class LayeredGraph:
    def __init__(self):
        self.G = []

    def gen_graph(self, depth=5, vertexs_size=20, connection_size=30):
        # 0 base
        # iter vertex number of each layer
        import random
        bucket = [1] * depth
        for _ in range(vertexs_size - depth):
            idx = random.randint(0, depth - 1)
            bucket[idx] += 1

        # iter vertex
        start_idx = 0
        self.G = []
        for layer, size in enumerate(bucket):
            for i in range(start_idx, start_idx + size):
                self.add_vertex(Vertex(i, layer))
            start_idx += size

        # random connect vertex
        for _ in range(connection_size):
            while True:
                start_v = self.G[random.randint(0, vertexs_size - 1)]
                end_v = self.G[random.randint(0, vertexs_size - 1)]

                if start_v.layer < end_v.layer and not Vertex.is_connected(start_v, end_v):
                    start_v.add_next(end_v.id_)
                    end_v.add_from(start_v.id_)
                    break

    def add_vertex(self, vertex):
        self.G += [vertex]

    def from_vertexs(self, vertex_id):
        return [self.G[vertex_id] for vertex_id in self.G[vertex_id].from_]


def merge_add(inputs, name="merge_add"):
    with tf.variable_scope(name):
        merged = None

        for input_ in inputs:
            if merged is None:
                merged = input_
            else:
                merged = tf.add(merged, input_)
        return merged


def layeredVertexBlock3(input_, layerGraph, name="layeredVertexBlock", output_size=10):
    # all vertex in and out
    # and conv

    with tf.variable_scope(name):
        for v in layerGraph.G:
            with tf.variable_scope("vertex" + str(v.id_)):
                # all first layer vertex connect to input_
                assigns = [conv_block(input_, output_size, filter_3311, relu)]

                # other layer collect from vertex, assign v_
                from_vertexs = layerGraph.from_vertexs(v.id_)
                for vertex in from_vertexs:
                    assign = conv_block(vertex.tensor, output_size, filter_5511, lrelu,
                                        name="from_assign" + str(vertex.id_))
                    assigns += [assign]

                v.tensor = merge_add(assigns)

        with tf.variable_scope("merge_concat"):
            out = tf.concat([vertex.tensor for vertex in layerGraph.G], axis=3)
            print(out)

    return out


def inception_layer(input_, channel_size, name='inception_layer'):
    with tf.variable_scope(name):
        with tf.variable_scope('out1'):
            out1 = tf.nn.avg_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        with tf.variable_scope('out2'):
            out2 = conv_block(input_, channel_size, filter_5511, lrelu, name='block1')

        with tf.variable_scope('out3'):
            out3 = conv_block(input_, channel_size, filter_5511, lrelu, name='block1')
            out3 = conv_block(out3, channel_size, filter_5511, lrelu, name='block2')

        with tf.variable_scope('out4'):
            out4 = conv_block(input_, channel_size, filter_5511, relu, name='block1')
            out4 = conv_block(out4, channel_size, filter_5511, relu, name='block2')
            out4 = conv_block(out4, channel_size, filter_5511, relu, name='block3')

        out = tf.concat([out1, tf.add(out2, out3)], 3)
        # out = tf.add(out2, out3, out4)

    return out


class WGAN1(AbstractGANModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def hyper_parameter(self):
        self.n_noise = 512
        self.batch_size = 64
        self.learning_rate = 0.0002

        self.beta1 = 0.5
        self.disc_iters = 1
        self.layeredGraph = LayeredGraph()
        depth = 2
        vertexs_size = depth * 8
        connection_size = int(vertexs_size * 2)
        self.layeredGraph.gen_graph(depth=depth, vertexs_size=vertexs_size, connection_size=connection_size)
        for v in self.layeredGraph.G:
            print(v)

    def generator(self, z, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            seq = SequenceModel(z)
            seq.add_layer(linear, 4 * 4 * 512)
            seq.add_layer(tf.reshape, [self.batch_size, 4, 4, 512])

            seq.add_layer(conv2d_transpose, [self.batch_size, 8, 8, 256], filter_5522)
            seq.add_layer(bn)
            seq.add_layer(lrelu)

            seq.add_layer(conv2d_transpose, [self.batch_size, 16, 16, 128], filter_5522)
            seq.add_layer(bn)
            seq.add_layer(lrelu)

            seq.add_layer(layeredVertexBlock3, self.layeredGraph, output_size=4)

            seq.add_layer(conv2d_transpose, [self.batch_size, 32, 32, self.input_c], filter_5522)

            seq.add_layer(conv2d, self.input_c, filter_5511)
            seq.add_layer(tf.sigmoid)
            net = seq.last_layer

        return net

    def discriminator(self, x, reuse=None, name='discriminator'):
        with tf.variable_scope(name, reuse=reuse):
            seq = SequenceModel(x)
            seq.add_layer(layeredVertexBlock3, self.layeredGraph, output_size=4)

            seq.add_layer(conv2d, 64, filter_5522)
            seq.add_layer(bn)
            seq.add_layer(lrelu)
            seq.add_layer(inception_layer, 32)
            seq.add_layer(inception_layer, 64)
            seq.add_layer(inception_layer, 128)

            seq.add_layer(tf.reshape, [self.batch_size, -1])
            out_logit = seq.add_layer(linear, 1)
            out = seq.add_layer(tf.sigmoid)

        return out, out_logit

    def network(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size] + self.shape_data_x, name='X')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.n_noise], name='z')

        self.G = self.generator(self.z)
        self.D_real, self.D_real_logit = self.discriminator(self.X)
        self.D_gen, self.D_gene_logit = self.discriminator(self.G, True)

    def loss(self):
        with tf.variable_scope('loss'):
            with tf.variable_scope('loss_D_real'):
                self.loss_D_real = -tf.reduce_mean(self.D_real)
            with tf.variable_scope('loss_D_gen'):
                self.loss_D_gen = tf.reduce_mean(self.D_gen)
            with tf.variable_scope('loss_D'):
                self.loss_D = self.loss_D_real + self.loss_D_gen
            with tf.variable_scope('loss_G'):
                self.loss_G = -self.loss_D_gen

    def train_ops(self):
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='discriminator')

        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='generator')

        with tf.variable_scope('train_ops'):
            self.train_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss_D, var_list=self.vars_D)
            self.train_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss_G, var_list=self.vars_G)

        with tf.variable_scope('clip_D_op'):
            self.clip_D_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.vars_D]

    def misc_ops(self):
        super().misc_ops()
        with tf.variable_scope('misc_op'):
            self.GD_rate = tf.div(tf.reduce_mean(self.loss_G), tf.reduce_mean(self.loss_D))

    def train_model(self, sess=None, iter_num=None, dataset=None):
        noise = self.get_noise()
        batch_xs = dataset.next_batch(self.batch_size, batch_keys=[BATCH_KEY_TRAIN_X])
        sess.run([self.train_D, self.clip_D_op], feed_dict={self.X: batch_xs, self.z: noise})

        if iter_num % self.disc_iters == 0:
            sess.run(self.train_G, feed_dict={self.z: noise})

        sess.run([self.op_inc_global_step])

    def summary_op(self):
        summary_loss(self.loss_D_gen)
        summary_loss(self.loss_D_real)
        summary_loss(self.loss_D)
        summary_loss(self.loss_G)

        self.op_merge_summary = tf.summary.merge_all()

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        noise = self.get_noise()
        batch_xs = dataset.next_batch(self.batch_size, batch_keys=[BATCH_KEY_TRAIN_X])
        summary, global_step = sess.run([self.op_merge_summary, self.global_step],
                                        feed_dict={self.X: batch_xs, self.z: noise})
        summary_writer.add_summary(summary, global_step)

    def get_noise(self):
        return np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.n_noise])
