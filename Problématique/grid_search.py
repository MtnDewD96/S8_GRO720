import itertools

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer


def create_network(checkpoint_path):
    layers = [
        FullyConnectedLayer(784,128), 
        BatchNormalization(128), 
        ReLU(), 
        FullyConnectedLayer(128,32), 
        BatchNormalization(32), 
        ReLU(), 
        FullyConnectedLayer(32,10)
    ]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    epoch_count = 50

    learning_rates = [0.1, 0.2, 0.4]
    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        print(f"Training for learning rate = {lr}, batch size = {bs} and epoch count = {epoch_count}")
        output_path = f"trainers/e{epoch_count}_a{lr:.4f}_b{bs}"
        network = create_network(None)
        trainer = MnistTrainer(network, lr, epoch_count, bs, output_path)
        trainer.train()