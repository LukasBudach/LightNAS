from bmxnet_examples.binary_models.meliusnet import meliusnet22
from autogluon import ImageClassification as task
from mxnet.gluon.data.vision.datasets import CIFAR10
from config import DATASET_PATH
from mxnet import nd
import mxnet as mx

def train_meliusnet(epochs):
    net = meliusnet22()

    #dataset = CIFAR10(DATASET_PATH+'/cifar10')
    #net.initialize()
    #t = nd.ones((2, 3, 32, 32))
    #net(t)
    task.fit(dataset='cifar10', net=net, epochs=epochs, ngpus_per_trial=1, batch_size=2)



def main():
    train_meliusnet(1)

if __name__=="__main__":
    main()