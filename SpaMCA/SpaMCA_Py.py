import math
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing, fix_seed


# =============================================================================
# SpatialGlue模型训练类
# =============================================================================
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
                 weight_factors=[0.1, 0.1, 0.1, 0.1],
                 ):
        """
        初始化函数，用于配置模型训练的基本参数

        参数：
        ----------
        data : dict
            包含空间多组学数据的字典对象
        datatype : string, optional
            输入数据类型，支持'SPOTS', 'Stereo-CITE-seq', 'Spatial-ATAC-RNA-seq'等，默认是'SPOTS'
        device : string, optional
            使用GPU还是CPU计算，默认是'cpu'
        random_seed : int, optional
            随机种子，用于固定模型初始化，默认是2022
        learning_rate : float, optional
            学习率，默认是0.001
        weight_decay : float, optional
            权重衰减参数，控制权重的影响，默认是0.00
        epochs : int, optional
            模型训练的轮数，默认是1500
        dim_input : int, optional
            输入特征的维度，默认是3000
        dim_output : int, optional
            输出表示的维度，默认是64
        weight_factors : list, optional
            平衡不同组学数据对模型训练影响的权重因子列表，默认是[1, 5, 1, 1]

        返回：
        -------
        self.emb_combined : tensor
            模型训练后得到的联合嵌入表示
        """

        # 将输入数据复制到实例变量中
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output

        # 获取两个组学的数据对象
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        # 构建邻接矩阵(空间和特征)
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # 转换特征为PyTorch张量并移动到指定设备
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        # 统计每个组学数据的细胞数量
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # 输入输出维度设置
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        # 根据不同的数据类型调整训练轮数和权重因子
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
        elif self.datatype == 'D1':
            self.epochs = 600
            self.n_clusters = 5
            self.mask_rate = 0
            self.weight_factors = (0.1, 0.7, 0.4, 0.2)
        elif self.datatype == 'A1':
            self.epochs = 600
            self.n_clusters = 10
            self.mask_rate = 0.2
            self.weight_factors = (0.1, 0.4, 0.1, 0.6)
        elif self.datatype == 'MTS':
            self.epochs = 500
            self.n_clusters = 5
            self.mask_rate = 0.5
            self.weight_factors = (0, 0.1, 0.1, 0.1)

        # 初始化损失函数
        self.instanceLoss = InstanceLoss(temperature=0.5, device=self.device)
        self.clusterLoss = ClusterLoss(temperature=0.5, class_num=self.n_clusters, device=self.device)

    def train(self):
        """
        模型训练主函数，构建编码器、定义优化器、进行多轮训练并返回最终的嵌入表示

        训练过程：
        - 定义模型结构 Encoder_overall
        - 使用 Adam 优化器
        - 每个 epoch 前向传播计算损失
        - 反向传播更新参数
        - 最终评估模式下获取嵌入结果
        - 归一化并返回联合嵌入表示
        """
        fix_seed(2022)
        # 初始化模型，并移动到指定设备
        self.model = Encoder_overall(
            self.dim_input1,
            self.dim_output1,
            self.dim_input2,
            self.dim_output2,
            self.n_clusters,
            self.mask_rate,
        ).to(self.device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 设置模型为训练模式
        self.model.train()


        # 开始训练循环
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.model.train()

            # 前向传播，得到模型输出
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )
            # 计算重建损失(MSE)
            indices_RNA = results['node_RNA']  # shape: (num_masked_nodes, )
            indices_ADT = results['node_ADT']
            self.loss_recon_omics1 = F.mse_loss(
                self.features_omics1[indices_RNA],
                results['emb_recon_omics1'][indices_RNA]
            )
            self.loss_recon_omics2 = F.mse_loss(
                self.features_omics2[indices_ADT],
                results['emb_recon_omics2'][indices_ADT]
            )

            # 最大概率为标签
            EPS = 1e-8

            # 从results中取出三个聚类输出
            y1 = results['emb_cluster1']
            y2 = results['emb_cluster2']
            y = results['emb_cluster']

            # 找出最大值作为目标分布
            y_max = torch.maximum(torch.maximum(y1, y2), y)

            # 行归一化(使每一行的和为1)
            y_max = y_max / (y_max.sum(dim=1, keepdim=True) + EPS)

            # 对原始y加一个最小值限制，防止log(0)
            y = torch.clamp(y, min=EPS)

            # 归一化y(如果还没归一化)
            y = y / (y.sum(dim=1, keepdim=True) + EPS)

            # 计算KL散度损失
            self.hc_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(y),
                y_max.detach()
            )

            # 实例损失(使用SCE损失)
            self.inst_loss = sce_loss(
                results['emb_instanceProject_omics1'],
                results['emb_instanceProject_omics2']
            )

            # 簇级损失
            self.clust_loss = self.clusterLoss(
                results['emb_cluster1'],
                results['emb_cluster2']
            )

            # 计算总损失(加权求和)
            loss = (
                    (self.loss_recon_omics1 + self.loss_recon_omics2) * self.weight_factors[0] +
                    self.weight_factors[1] * self.inst_loss +
                    self.weight_factors[2] * self.clust_loss +
                    self.hc_loss * self.weight_factors[3]
            )

            # 更新进度条信息
            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                recon1=f"{self.loss_recon_omics1.item():.4f}",
                recon2=f"{self.loss_recon_omics2.item():.4f}",
                hc=f"{self.hc_loss.item():.4f}",
                inst=f"{self.inst_loss.item():.4f}",
                clust=f"{self.clust_loss.item():.4f}"
            )
            pbar.update(1)

            # 清空梯度
            self.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()

        print("Model training finished!\n")

        # 测试模式下不计算梯度
        with torch.no_grad():
            self.model.eval()
            # 再次前向传播获取最终结果
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )

        # 对最终嵌入进行L2归一化
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        # 准备输出结果
        output = {
            'SpaMCA': emb_combined.detach().cpu().numpy(),
        }
        return output


# 多用于联合训练潜在空间对齐任务(如omics1和omics2的一致性约束)
def sce_loss(emb1, emb2, t=2):
    """
    对称对比嵌入损失(SCE)，用于对齐两个嵌入空间

    参数:
        emb1 (torch.Tensor): 第一个嵌入张量 [batch_size, dim]
        emb2 (torch.Tensor): 第二个嵌入张量 [batch_size, dim]
        t (float): 温度参数，用于缩放

    返回:
        torch.Tensor: 计算得到的SCE损失
    """
    # L2归一化
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)

    # 计算余弦相似度，并映射到[0,1]范围
    cos_sim = (emb1 * emb2).sum(dim=-1)  # 点积计算相似度
    cos_m = (cos_sim + 1.0) * 0.5  # 将[-1,1]映射到[0,1]

    # 温度缩放 + 负对数似然
    loss = -torch.log(cos_m.pow(t))  # 温度缩放后的负对数似然

    return loss.mean()  # 返回平均损失

class InstanceLoss(nn.Module):
    """
    实例级别的对比损失，用于学习判别性特征
    """

    def __init__(self, temperature=0.5, device=None):
        """
        初始化实例损失

        参数:
        ----------
        temperature : float
            温度参数
        device : torch.device
            计算设备
        """
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        """
        根据当前batch_size构造排除正样本对的mask

        参数:
        ----------
        batch_size : int
            批大小

        返回:
        -------
        torch.Tensor
            布尔型掩码张量
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(0)  # 排除自身
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        计算实例对比损失

        参数:
        ----------
        z_i : torch.Tensor
            第一个视图的嵌入向量 [batch_size, embedding_dim]
        z_j : torch.Tensor
            第二个视图的嵌入向量 [batch_size, embedding_dim]

        返回:
        -------
        torch.Tensor
            对比损失值
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # 拼接两个视图的表示
        z = torch.cat([z_i, z_j], dim=0)

        # 计算相似度矩阵
        sim = torch.mm(z, z.T) / self.temperature

        # 提取正样本对角线元素
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).view(N, 1)

        # 构建负样本(排除正样本和自身)
        mask = self.mask_correlated_samples(batch_size).to(sim.device)
        negative_samples = sim[mask].view(N, -1)

        # 合并正负样本logits
        logits = torch.cat([positive_samples, negative_samples], dim=1)

        # 分类目标:第一个位置是正例
        labels = torch.zeros(N, device=logits.device).long()

        # 计算损失
        loss = self.criterion(logits, labels) / N

        return loss


# 类簇级别的对比损失:用于鼓励不同视图中相同类别在聚类空间中靠近
class ClusterLoss(nn.Module):
    """
    类簇级别的对比损失，用于对齐不同视图中的类簇分布
    """

    def __init__(self, class_num, temperature, device):
        """
        初始化类簇损失

        参数:
        ----------
        class_num : int
            类簇/类别数量
        temperature : float
            温度参数
        device : torch.device
            计算设备
        """
        super(ClusterLoss, self).__init__()
        self.class_num = class_num  # 聚类数量(类别数)
        self.temperature = temperature  # 温度系数，控制分布锐利程度
        self.device = device  # 设备(CPU/GPU)

        self.mask = self.mask_correlated_clusters(class_num)  # 构造负样本掩码
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # 分类损失函数
        self.similarity_f = nn.CosineSimilarity(dim=2)  # 余弦相似度计算

    def mask_correlated_clusters(self, class_num):
        """
        构建一个掩码矩阵，屏蔽掉正样本对和对角线元素

        参数:
        ----------
        class_num : int
            类簇数量

        返回:
        -------
        torch.Tensor
            布尔型掩码张量
        """
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)  # 屏蔽对角线(自己和自己不是负样本)
        for i in range(class_num):
            mask[i, class_num + i] = 0  # 屏蔽正样本对
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        """
        计算类簇对比损失

        参数:
        ----------
        c_i : torch.Tensor
            第一个视图的聚类概率分布 [batch_size, num_classes]
        c_j : torch.Tensor
            第二个视图的聚类概率分布 [batch_size, num_classes]
        alpha : float
            正则化系数，控制熵最小化项的权重

        返回:
        -------
        torch.Tensor
            计算得到的类簇损失
        """
        # 计算每个聚类的概率分布p_i和p_j
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()  # 负熵项

        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        ne_loss = ne_i + ne_j  # 总熵损失

        # 转置以适应后续计算
        c_i = c_i.t()
        c_j = c_j.t()

        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)  # 合并两个视图的聚类结果

        # 计算余弦相似度矩阵，并除以温度系数
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        # 提取正样本对(i <-> j)
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        # 正样本对拼接
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # 负样本对提取
        negative_clusters = sim[self.mask].reshape(N, -1)

        # 标签为0，表示只有第一个位置是正样本
        labels = torch.zeros(N).to(positive_clusters.device).long()

        # 拼接logits
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)

        # 计算交叉熵损失
        loss = self.criterion(logits, labels)
        loss /= N  # 归一化

        return loss + alpha * ne_loss  # 返回总损失