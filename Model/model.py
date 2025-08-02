import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1,
                 dim_in_feat_omics2, dim_out_feat_omics2, n_clusters,mask_rate,
                 dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()

        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.act = act
        self.mask_rate = mask_rate

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1,self.mask_rate)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2,self.mask_rate)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)

        self.instanceProject1 = InstanceProject(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.instanceProject2 = InstanceProject(self.dim_out_feat_omics2,
                                                self.dim_out_feat_omics1)
        self.MLP = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, self.dim_out_feat_omics1)

        self.clusters_1 = ClusterProject(self.dim_out_feat_omics1, self.n_clusters)
        self.clusters_2 = ClusterProject(self.dim_out_feat_omics2, self.n_clusters)
        self.clusters_3 = ClusterProject(self.dim_out_feat_omics2, self.n_clusters)

    def forward(self, features_omics1, features_omics2,
                adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2):
        emb_latent_spatial_omics1, node_RNA = self.encoder_omics1(features_omics1, adj_spatial_omics1)

        emb_latent_feature_omics1, _ = self.encoder_omics1(features_omics1, adj_feature_omics1)

        emb_latent_spatial_omics2, node_ADT = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        emb_latent_feature_omics2, _ = self.encoder_omics2(features_omics2, adj_feature_omics2)

        emb_latent_omics1, alpha_omics1 = self.atten_omics1(
            emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(
            emb_latent_spatial_omics2, emb_latent_feature_omics2)

        cat_emb_latent = torch.cat((emb_latent_omics1, emb_latent_omics2), dim=1)
        emb_latent_combined = self.MLP(cat_emb_latent)

        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        emb_instanceProject_omics1 = self.instanceProject1(emb_latent_omics1)
        emb_instanceProject_omics2 = self.instanceProject2(emb_latent_omics2)

        emb_cluster1 = self.clusters_1(emb_latent_omics1)
        emb_cluster2 = self.clusters_2(emb_latent_omics2)
        emb_cluster = self.clusters_3(emb_latent_combined)

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_cluster1': emb_cluster1,
            'emb_cluster2': emb_cluster2,
            'emb_cluster': emb_cluster,
            'emb_instanceProject_omics1': emb_instanceProject_omics1,
            'emb_instanceProject_omics2': emb_instanceProject_omics2,
            'node_RNA': node_RNA,
            'node_ADT': node_ADT
        }
        return results


class Encoder(Module):
    def __init__(self, in_feat, out_feat,mask_rate, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.mask_rate = mask_rate
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def encoding_mask_noise(self, x, adj):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] = 0.0

        use_adj = adj.clone()
        return out_x, use_adj, (mask_nodes, keep_nodes)

    def forward(self, feat, adj):
        if self.mask_rate == 0 :
            x = torch.mm(feat, self.weight)
            x = torch.spmm(adj, x)
            return x,feat.shape[0]-1

        out_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, adj)
        x = torch.mm(out_x, self.weight)
        x = torch.spmm(use_adj, x)
        return x, mask_nodes


class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class CrossAttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(CrossAttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_query = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.w_key = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.w_value = Parameter(torch.FloatTensor(in_feat, out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_query)
        torch.nn.init.xavier_uniform_(self.w_key)
        torch.nn.init.xavier_uniform_(self.w_value)

    def forward(self, emb1, emb2):
        emb1 = emb1.unsqueeze(0)
        emb2 = emb2.unsqueeze(0)

        query = torch.matmul(emb1, self.w_query)
        key = torch.matmul(emb2, self.w_key)
        value = torch.matmul(emb2, self.w_value)

        energy = torch.bmm(query, key.transpose(1, 2))
        attention = F.softmax(energy, dim=-1)

        output = torch.bmm(attention, value)

        output = output.squeeze(0)

        return output, attention


class InstanceProject(nn.Module):
    def __init__(self, dim1, dim2):
        super(InstanceProject, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        self.instance_projector = nn.Sequential(
            nn.Linear(self.dim1, self.dim2),
            nn.BatchNorm1d(self.dim2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.instance_projector(x)


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()

        self._latent_dim = latent_dim
        self._n_clusters = n_clusters

        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )

        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)

        y = self.cluster(z)

        return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out