from model.AbstractModel import AbstractModel
from util.SequenceModel import SequenceModel
from util.ops import *
from util.summary_func import *
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *


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


def layeredVertexBlock3(input_, tensor_op, layerGraph, name="layeredVertexBlock", output_size=10):
    # all vertex in and out
    # and conv
    op = tensor_op

    # net, output_channel, filter_, activate, name = 'conv_block'):
    with tf.variable_scope(name):
        for v in layerGraph.G:
            print(v)
            with tf.variable_scope("vertex" + str(v.id_)):
                # all first layer vertex connect to input_
                assigns = [conv_block(input_, output_size, filter_3311, relu)]

                # other layer collect from vertex, assign v_
                from_vertexs = layerGraph.from_vertexs(v.id_)
                for vertex in from_vertexs:
                    assign = conv_block(vertex.tensor, output_size, filter_3311, relu,
                                        name="from_assign" + str(vertex.id_))
                    assigns += [assign]

                for assign in assigns:
                    print(assign)
                v.tensor = merge_add(assigns)
                print(v)

        with tf.variable_scope("merge_concat"):
            out = tf.concat([vertex.tensor for vertex in layerGraph.G], axis=1)
            print(out)

    return out


class LayeredVertexGraph8(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __str__(self):
        return "Classifier"

    def hyper_parameter(self):
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.beta1 = 0.5

    def input_shapes(self, input_shapes):
        shape_data_x = input_shapes[INPUT_SHAPE_KEY_DATA_X]
        if len(shape_data_x) == 3:
            self.shape_data_x = shape_data_x
            H, W, C = shape_data_x
            self.input_size = W * H * C
            self.input_w = W
            self.input_h = H
            self.input_c = C
        elif len(shape_data_x) == 2:
            self.shape_data_x = shape_data_x + [1]
            H, W = shape_data_x
            self.input_size = W * H
            self.input_w = W
            self.input_h = H
            self.input_c = 1

        self.label_shape = input_shapes[INPUT_SHAPE_KEY_LABEL]
        self.label_size = input_shapes[INPUT_SHAPE_KEY_LABEL_SIZE]

    def classifier(self, input_):

        with tf.variable_scope('classifier'):
            seq = SequenceModel(input_)

            # seq.add_layer(tf.reshape, [self.batch_size, -1])

            layeredGraph = LayeredGraph()
            output_size = 20
            depth = 5
            vertexs_size = depth * 10
            connection_size = int(vertexs_size * 4)
            layeredGraph.gen_graph(depth=depth, vertexs_size=vertexs_size, connection_size=connection_size)
            for v in layeredGraph.G:
                print(v)
            seq.add_layer(layeredVertexBlock3, None, layeredGraph, output_size=output_size)
            seq.add_layer(tf.reshape, [self.batch_size, -1])

            seq.add_layer(linear, self.label_size)
            logit = seq.last_layer
            h = softmax(logit)

        return logit, h

    def network(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size] + self.shape_data_x, name='X')
        self.label = tf.placeholder(tf.float32, [self.batch_size] + self.label_shape, name='label')

        self.logit, self.h = self.classifier(self.X)

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.label)
        self.batch_acc = tf.reduce_mean(tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64),
                                        name="batch_acc")

    def loss(self):
        with tf.variable_scope('loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logit)
            self.loss_mean = tf.reduce_mean(self.loss)

    def train_ops(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')

        with tf.variable_scope('train_ops'):
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss, var_list=self.vars)

        with tf.variable_scope('clip_op'):
            self.clip_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.vars]

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs, batch_labels = dataset.next_batch(self.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        sess.run([self.train, self.clip_op], feed_dict={self.X: batch_xs, self.label: batch_labels})

        sess.run([self.op_inc_global_step])

    def summary_op(self):
        summary_loss(self.loss)
        summary_variable_mean(self.batch_acc, "test_acc")

        self.op_merge_summary = tf.summary.merge_all()

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        batch_xs, batch_labels = dataset.next_batch(self.batch_size,
                                                    batch_keys=[BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL])
        summary, global_step = sess.run([self.op_merge_summary, self.global_step],
                                        feed_dict={self.X: batch_xs, self.label: batch_labels})
        summary_writer.add_summary(summary, global_step)
