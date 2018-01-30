from workbench.InstanceManagerHelper import InstanceManagerHelper
from env_settting import *

# TODO dynamic module load class or function for import model and visualizer
from visualizer.image_tile import image_tile
from visualizer.log_D_value import print_D_value
from visualizer.log_gan_loss import print_GAN_loss
from visualizer.image_tile_data import image_tile_data
from visualizer.log_confusion_matrix import print_confusion_matrix
from visualizer.log_classifier_loss import print_classifier_loss

# dataset
from workbench.MNISTHelper import MNISTHelper
from workbench.fashion_MNISTHelper import fashion_MNISTHelper
from workbench.cifar10Helper import cifar10Helper
from workbench.cifar100Helper import cifar100Helper
from workbench.MNISTHelper import MNISTHelper
from workbench.LLDHelper import LLD_helper


# TODO NHWC format is right not NWHC
# NWHC = Num_samples x Height x Width x Channels

def load_model_class_from_module(module_path, class_name):
    from util.util import load_class_from_source_path
    from glob import glob

    model = None
    paths = glob(os.path.join(module_path, '**', '*.py'), recursive=True)
    for path in paths:
        file_name = os.path.basename(path)
        if file_name == class_name + '.py':
            model = load_class_from_source_path(path, class_name)
            print('load class %s from %s' % (model, path))
    del load_class_from_source_path
    del glob
    if model is None:
        print("model class '%s' not found" % class_name)

    return model


def add_all(inputs):
    # return add inputs
    return None


def gen_meta_subgraph(input_, depth=5, subnet_num=20, subnet_size=5, connection_num=30):
    subG = [[] for _ in range(depth)]
    # 0's layer for input_
    subG[0] += [input_]

    import random

    # set sub_net at layer
    for i in range(subnet_num):
        layer_num = random.randint(1, depth)
        subG[layer_num] += [i]
    print(subG)

    # connect sub_net
    # for i in range(connection_num):
    #     # select random start, end vertex
    #     start_layer = random.randint(1, depth - 1)
    #     end_layer = random.randint(start_layer + 1, depth)
    #     assert (start_layer < end_layer)
    #     start_v_num = random.randint
    #
    #     # connect


def main():
    gen_meta_subgraph(None)


    # import env_settting
    # env_settting.tensorboard_dir()
    #
    # # dataset, input_shapes = cifar10Helper.load_dataset(limit=1000)
    # # dataset, input_shapes = MNISTHelper.load_dataset(limit=1000)
    # # dataset, input_shapes = cifar100Helper.load_dataset(limit=1000)
    # # dataset, input_shapes = fashion_MNISTHelper.load_dataset(limit=1000)
    # # dataset, input_shapes = LLD_helper.load_dataset(limit=1000)
    #
    # dataset, input_shapes = MNISTHelper.load_dataset()
    # visualizers = [(print_classifier_loss, 10), (print_confusion_matrix, 50)]
    # model = load_model_class_from_module(MODEL_MODULE_PATH, 'Classifier')
    # InstanceManagerHelper.gen_model_and_train(model=model,
    #                                           input_shapes=input_shapes,
    #                                           visualizers=visualizers,
    #                                           env_path=ROOT_PATH,
    #                                           dataset=dataset,
    #                                           epoch_time=5)
