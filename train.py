import os
import sys
import time
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import models
from arguments import arguments
from utils import prepare_data, kl_anneal_function


def run_epoch(dataloader, train):
    model.train() if train else model.eval()

    epoch_loss, epoch_recon_c, epoch_recon_b, epoch_kld = 0., 0., 0., 0.
    for x_c, m_c, x_b, m_b, _ in dataloader:
        model.zero_grad()

        x_c = x_c.to(device)
        m_c = m_c.to(device)
        x_b = x_b.to(device)
        m_b = m_b.to(device)

        mu, log_var, decoder_output = model(x_c, m_c, x_b, m_b)
        recon_c, recon_b, kld = model.calculate_loss(mu, log_var, x_c, m_c, x_b, m_b, decoder_output)
        loss = recon_c + recon_b + beta * kld

        epoch_loss += loss.item()
        epoch_recon_c += recon_c.item()
        epoch_recon_b += recon_b.item()
        epoch_kld += kld.item()

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    epoch_loss /= len(dataloader)
    epoch_recon_c /= len(dataloader)
    epoch_recon_b /= len(dataloader)
    epoch_kld /= len(dataloader)

    losses = {"loss": epoch_loss,
              "recon_c": epoch_recon_c,
              "recon_b": epoch_recon_b,
              "kld": epoch_kld}
    return losses


if __name__ == "__main__":
    args = arguments(sys.argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(args)
    for i, d in enumerate(data["data_cv"]):
        np.random.seed(42+i)
        torch.manual_seed(42+i)

        writer = SummaryWriter(os.path.join(args.tensorboard_path, f"cv{i}"))
        model = models.VAE(input_size_c=len(args.c_cols),
                           input_size_b=len(args.b_cols),
                           hidden_size_c=args.hidden_size_c,
                           hidden_size_b=args.hidden_size_b,
                           hidden_size_d=args.hidden_size_d,
                           latent_size=args.latent_size,
                           num_layers=args.num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        beta = 0.
        for epoch in range(1, args.epoch+1):
            s = time.time()
            epoch_losses_train = run_epoch(d["dataloader_train"], True)
            epoch_losses_test = run_epoch(d["dataloader_val"], False)
            beta = kl_anneal_function(epoch, 1/128, 1000, 1/8)

            writer.add_scalar("train/loss", epoch_losses_train["loss"], epoch)
            writer.add_scalar("train/recon_c", epoch_losses_train["recon_c"], epoch)
            writer.add_scalar("train/recon_b", epoch_losses_train["recon_b"], epoch)
            writer.add_scalar("train/kld", epoch_losses_train["kld"], epoch)
            writer.add_scalar("test/loss", epoch_losses_test["loss"], epoch)
            writer.add_scalar("test/recon_c", epoch_losses_test["recon_c"], epoch)
            writer.add_scalar("test/recon_b", epoch_losses_test["recon_b"], epoch)
            writer.add_scalar("test/kld", epoch_losses_test["kld"], epoch)

            e = time.time()
            t = f"epoch: {epoch:04d} " + \
                f"train: {epoch_losses_train['loss']:.4f}" + \
                f"/{epoch_losses_train['recon_c']:.4f}" + \
                f"/{epoch_losses_train['recon_b']:.4f}" + \
                f"/{epoch_losses_train['kld']:.4f} " + \
                f"test: {epoch_losses_test['loss']:.4f}"+ \
                f"/{epoch_losses_test['recon_c']:.4f}"+ \
                f"/{epoch_losses_test['recon_b']:.4f}"+ \
                f"/{epoch_losses_test['kld']:.4f} " + \
                f"time: {int(e-s)} sec " + \
                f"beta: {beta:.4f}"
            print(t)

