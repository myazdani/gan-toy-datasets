import torch
from torch import nn
from torch import optim
import pandas as pd


class GAN:
    def __init__(self, dim, z_dim, nb_hidden=100, lr=1e-3):
        self.model_G = nn.Sequential(
            nn.Linear(z_dim, nb_hidden), nn.ReLU(), nn.Linear(nb_hidden, dim)
        )

        self.model_D = nn.Sequential(
            nn.Linear(dim, nb_hidden), nn.ReLU(), nn.Linear(nb_hidden, 1), nn.Sigmoid()
        )

        self.dim = dim
        self.z_dim = z_dim

        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr)

    def generator_loss(self, fake_scores, real_scores=None):
        loss = (1 - fake_scores).log().mean()
        return loss

    def discriminator_loss(self, fake_scores, real_scores):
        loss = -(1 - fake_scores).log().mean() - real_scores.log().mean()
        return loss

    def eval_discriminator(self, real_batch, real_samples, exp_name):
        z = real_batch.new(real_samples.size(0), self.z_dim).normal_()
        fakes = self.model_G(z)
        fake_preds = self.model_D(fakes)
        real_preds = self.model_D(real_samples)
        fake_discs = fake_preds.mean().detach().numpy()
        real_discs = real_preds.mean().detach().numpy()
        fake_samples = fakes.detach().numpy()
        df_fake_samples = pd.DataFrame(fake_samples, columns=["x", "y"])
        df_fake_samples["exp_name"] = exp_name
        df_fake_samples.to_csv(exp_name + "_fake_samples_" + ".csv", index=None)
        return fake_discs, real_discs

    def train(self, real_samples, nb_epochs, batch_size, exp_name="", save_rate=10):
        fake_discs = []
        real_discs = []
        for epoch in range(nb_epochs):
            for t, real_batch in enumerate(real_samples.split(batch_size)):
                z = real_batch.new(real_batch.size(0), self.z_dim).normal_()
                fake_batch = self.model_G(z)
                D_scores_on_real = self.model_D(real_batch)
                D_scores_on_fake = self.model_D(fake_batch)
                if t % 2 == 0:
                    loss = self.generator_loss(D_scores_on_fake)
                    self.optimizer_G.zero_grad()
                    loss.backward()
                    self.optimizer_G.step()
                else:
                    loss = self.discriminator_loss(D_scores_on_fake, D_scores_on_real)
                    self.optimizer_D.zero_grad()
                    loss.backward()
                    self.optimizer_D.step()

            # Eval generator using the disriminator:
            #   generate some fake data and see how well discriminator can
            #   discriminate against a batch of real data
            # this is to save snapshot of GAN training throughout iterations
            if epoch % save_rate == 0 or epoch == nb_epochs - 1:
                print(epoch)
                (fake_batch_discs, real_batch_discs) = self.eval_discriminator(
                    real_batch, real_samples, exp_name = exp_name + "_" + str(epoch)
                )
                fake_discs.append(fake_batch_discs)
                real_discs.append(real_batch_discs)

        return fake_discs, real_discs

