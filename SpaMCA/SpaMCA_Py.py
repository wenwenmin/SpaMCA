import math
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing, fix_seed


class Train_SpaMCA:

    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2025,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=[0.1, 0.4, 0.1, 0.6],
                 ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        if self.datatype == 'SL':
            self.epochs = 1000
            self.n_clusters = 5
            self.mask_rate = 0.2
            self.weight_factors = (0.1, 0.1, 0.1, 0.2)
        elif self.datatype == 'ME':
            self.epochs = 500
            self.n_clusters = 14
            self.weight_factors = (0.1, 0.1, 0.7, 1.0)
            self.mask_rate = 0.2
        elif self.datatype == 'A1':
            self.epochs = 600
            self.n_clusters = 10
            self.mask_rate = 0.2
            self.weight_factors = (0.1, 0.4, 0.1, 0.6)
        self.instanceLoss = InstanceLoss(temperature=0.5, device=self.device)
        self.clusterLoss = ClusterLoss(temperature=0.5, class_num=self.n_clusters, device=self.device)

    def train(self):
        fix_seed(2022)
        self.model = Encoder_overall(
            self.dim_input1,
            self.dim_output1,
            self.dim_input2,
            self.dim_output2,
            self.n_clusters,
            self.mask_rate,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()

            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )
            indices_RNA = results['node_RNA']
            indices_ADT = results['node_ADT']
            self.loss_recon_omics1 = F.mse_loss(
                self.features_omics1[indices_RNA],
                results['emb_recon_omics1'][indices_RNA]
            )
            self.loss_recon_omics2 = F.mse_loss(
                self.features_omics2[indices_ADT],
                results['emb_recon_omics2'][indices_ADT]
            )

            EPS = 1e-8

            y1 = results['emb_cluster1']
            y2 = results['emb_cluster2']
            y = results['emb_cluster']

            y_max = torch.maximum(torch.maximum(y1, y2), y)
            
            p = torch.nn.functional.normalize(y_max ** 2, dim=1, p=2)

            p = p + EPS

            self.hc_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(y),
                p.detach()
            )

            self.inst_loss = sce_loss(
                results['emb_instanceProject_omics1'],
                results['emb_instanceProject_omics2']
            )

            self.clust_loss = self.clusterLoss(
                results['emb_cluster1'],
                results['emb_cluster2']
            )

            loss = (
                    (self.loss_recon_omics1 + self.loss_recon_omics2) * self.weight_factors[0] +
                    self.weight_factors[1] * self.inst_loss +
                    self.weight_factors[2] * self.clust_loss +
                    self.hc_loss * self.weight_factors[3]
            )

            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                recon1=f"{self.loss_recon_omics1.item():.4f}",
                recon2=f"{self.loss_recon_omics2.item():.4f}",
                hc=f"{self.hc_loss.item():.4f}",
                inst=f"{self.inst_loss.item():.4f}",
                clust=f"{self.clust_loss.item():.4f}"
            )
            pbar.update(1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'SpaMCA': emb_combined.detach().cpu().numpy(),
        }
        return output


def sce_loss(emb1, emb2, t=2):
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)

    cos_sim = (emb1 * emb2).sum(dim=-1)
    cos_m = (cos_sim + 1.0) * 0.5

    loss = -torch.log(cos_m.pow(t))

    return loss.mean()

class InstanceLoss(nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.mm(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).view(N, 1)

        mask = self.mask_correlated_samples(batch_size).to(sim.device)
        negative_samples = sim[mask].view(N, -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1)

        labels = torch.zeros(N, device=logits.device).long()

        loss = self.criterion(logits, labels) / N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()

        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()

        logits = torch.cat((positive_clusters, negative_clusters), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss

