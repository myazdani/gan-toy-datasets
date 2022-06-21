from dataclasses import dataclass
from typing import Dict
import itertools
from sklearn import datasets
import pandas as pd
import numpy as np

import torch
from gan_defs import GAN


def generate_datasets(n_samples=1500) -> Dict[str, np.ndarray]:
    generated_datasets = {}
    generated_datasets["circles"] = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05
    )[0]
    generated_datasets["moons"] = datasets.make_moons(n_samples=n_samples, noise=0.05)[
        0
    ]
    generated_datasets["blobs"] = datasets.make_blobs(
        n_samples=n_samples, random_state=8
    )[0]

    return generated_datasets


def get_exp_name(base_name, dataset_name, z_dim, nb_hidden, lr, nb_epochs, batch_size):
    exp_name = base_name
    exp_name += dataset_name
    exp_name += "_nbhidden:" + str(nb_hidden)
    exp_name += "_zdim:" + str(z_dim)
    exp_name += "_batchsize:" + str(batch_size)
    exp_name += "_lr:" + str(lr)

    return exp_name


def run_experiment(
    datasets, z_dim, nb_hidden, lr, nb_epochs, save_rate, batch_size, exp_name=""
):
    for dataset_name, dataset in datasets.items():

        exp_id = get_exp_name(
            exp_name, dataset_name, z_dim, nb_hidden, lr, nb_epochs, batch_size
        )

        gan = GAN(dim=2, z_dim=z_dim, nb_hidden=nb_hidden, lr=lr)
        real_samples = torch.from_numpy(dataset).float()
        fake_discs, real_discs = gan.train(
            real_samples,
            nb_epochs=nb_epochs,
            save_rate=save_rate,
            batch_size=batch_size,
            exp_name="./results/" + exp_id,
        )


def main():
    toy_datasets = generate_datasets()
    # z_dims = [1, 2, 4, 8]
    # nb_hiddens = [32, 64, 128, 256, 512]
    # batch_sizes = [4, 8, 16, 32]
    z_dims = [1, 2]
    nb_hiddens = [32, 64]
    batch_sizes = [4, 8]
    for z_dim, nb_hidden, batch_size in itertools.product(
        z_dims, nb_hiddens, batch_sizes
    ):
        run_experiment(
            toy_datasets,
            z_dim=z_dim,
            nb_hidden=nb_hidden,
            batch_size=batch_size,
            lr=1e-3,
            nb_epochs=20,
            save_rate=10,
        )


if __name__ == "__main__":
    main()

