from visualizer.AbstractPrintLog import AbstractPrintLog
from dict_keys.dataset_batch_keys import *
from queue import deque


class window:
    def __init__(self, window_size):
        self.window = deque()
        self.window_size = window_size

    def __len__(self):
        return len(self.window)

    def __repr__(self):
        return self.__class__.__name__

    def avg(self):
        return float(sum(self.window)) / float(len(self.window))

    def push(self, val):
        if len(self.window) == self.window_size:
            self.window.popleft()

        self.window.append(val)


class print_classifier_loss(AbstractPrintLog):

    def __init__(self, path=None, iter_cycle=None, name=None):
        super().__init__(path, iter_cycle, name)
        self.train_windows = window(50)
        self.test_windows = window(50)
        self.window_size = 50

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        batch_xs, batch_labels = dataset.next_batch(model.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        loss, train_acc, global_step = sess.run([model.loss_mean, model.batch_acc, model.global_step],
                                                feed_dict={model.X: batch_xs, model.label: batch_labels})

        batch_xs, batch_labels = dataset.next_batch(model.batch_size,
                                                    batch_keys=[BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL])
        test_acc = sess.run(model.batch_acc, feed_dict={model.X: batch_xs, model.label: batch_labels})

        self.train_windows.push(train_acc)
        self.test_windows.push(test_acc)

        self.log('global_step : %04d ' % global_step,
                 'loss: {:.4} '.format(loss),
                 'train acc: {:.4} '.format(train_acc),
                 'test acc: {:.4} '.format(test_acc),
                 'smooth train acc: {:.4} '.format(self.train_windows.avg()),
                 'smooth test acc: {:.4} '.format(self.test_windows.avg()),
                 )
