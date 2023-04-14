import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,
                 input_size_c,
                 input_size_b,
                 hidden_size_c,
                 hidden_size_b,
                 hidden_size_d,
                 latent_size,
                 num_layers):
        super(VAE, self).__init__()
        self.encoder_c = Encoder(input_size_c,
                                 hidden_size_c,
                                 num_layers)
        self.encoder_b = Encoder(input_size_b,
                                 hidden_size_b,
                                 num_layers)

        self.fc_mixer = nn.Linear(hidden_size_c+hidden_size_b, latent_size)
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_log_var = nn.Linear(latent_size, latent_size)

        self.decoder = Decoder(input_size_c,
                               input_size_b,
                               hidden_size_d,
                               latent_size,
                               num_layers)

    def forward(self, x_c, m_c, x_b, m_b):
        sequence_length = x_c.size(1)

        x_enc_c = self.encoder_c(x_c, m_c)
        x_enc_b = self.encoder_b(x_b, m_b)

        x = torch.concat([x_enc_c, x_enc_b], dim=-1)
        x = F.relu(self.fc_mixer(x))

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparametrize(mu, log_var)

        decoder_output = self.decoder(x_c, x_b, m_c, m_b, z)
        return mu, log_var, decoder_output

    def reparametrize(self, mu, log_var):
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def calculate_loss(mu, log_var,
                       x_c, m_c,
                       x_b, m_b,
                       decoder_output):
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)
        kl_divergence = torch.mean(kl_divergence)
    
        dist_gamma = GammaDistribution(concentration=decoder_output["alpha"],
                                       rate=decoder_output["beta"])
        recon_c = -dist_gamma.log_prob(x_c)
        recon_c = torch.clamp(recon_c, min=-10, max=10)
        recon_c = torch.nansum(recon_c * m_c) / torch.sum(m_c)
    
        recon_b = F.binary_cross_entropy_with_logits(decoder_output["bernoulli"], x_b, reduction="none")
        recon_b = torch.clamp(recon_b, min=-10, max=10)
        recon_b = torch.nansum(recon_b * m_b) / torch.sum(m_b)
    
        return recon_c, recon_b, kl_divergence


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_size*2, hidden_size)
        self.rnn_c = nn.LSTM(hidden_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)

    def forward(self, x, m):
        x = torch.concat([x, m], dim=-1)
        x = F.relu(self.embedding(x))
        x, _ = self.rnn_c(x)
        return x[:, -1, :]


class Decoder(nn.Module):
    def __init__(self,
                 input_size_c,
                 input_size_b,
                 hidden_size_d,
                 latent_size,
                 num_layers):
        super(Decoder, self).__init__()
        self.expand_z = nn.Linear(latent_size, hidden_size_d)
        self.embedding = nn.Linear((input_size_c+input_size_b)*2, hidden_size_d)
        self.rnn_c = nn.LSTM(hidden_size_d,
                             hidden_size_d,
                             num_layers,
                             batch_first=True)
        self.fc_alpha = nn.Linear(hidden_size_d, input_size_c)
        self.fc_beta = nn.Linear(hidden_size_d, input_size_c)
        self.softplus = nn.Softplus()
        self.fc_bernoulli = nn.Linear(hidden_size_d, input_size_b)

    def forward(self, x_c, x_b, m_c, m_b, z):
        x_c = torch.flip(x_c, dims=[1])
        x_b = torch.flip(x_b, dims=[1])
        m_c = torch.flip(m_c, dims=[1])
        m_b = torch.flip(m_b, dims=[1])

        x = torch.concat([x_c, x_b, m_c, m_b], axis=-1)
        x_init = torch.zeros([x.shape[0], 1, x.shape[2]]).to(z.device)
        x = torch.concat([x_init, x[:, :-1, :]], axis=1)
        x = self.embedding(x)

        z = z.unsqueeze(0)
        z = self.expand_z(z)
        c = torch.zeros(*z.shape).to(z.device)

        x, _ = self.rnn_c(x, (z, c))

        alpha = self.softplus(self.fc_alpha(x))
        beta = self.softplus(self.fc_beta(x))
        bernoulli = self.fc_bernoulli(x)
        
        alpha = torch.flip(alpha, dims=[1])
        beta = torch.flip(beta, dims=[1])
        bernoulli = torch.flip(bernoulli, dims=[1])

        decoder_output = {"alpha": alpha,
                          "beta": beta,
                          "bernoulli": bernoulli}
        return decoder_output


class GammaDistribution(torch.distributions.Distribution):
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate
    
    def log_prob(self, x, eps=0.):
        return (torch.xlogy(self.concentration, self.rate + eps) +
                torch.xlogy(self.concentration - 1, x + eps) -
                self.rate * x - torch.lgamma(self.concentration + eps))

