from torch.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import meshio
import time
from scipy.interpolate import griddata
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from GaussIntegral import GaussIntegral


class Loss:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        pass
        
    def loss_function(self, data_idx, Tetra_coord: torch.Tensor, Dir_Triangle_coord: torch.Tensor, Pre_Triangle_coord: torch.Tensor, Sym_Triangle_coord: torch.Tensor, Pre_load: float, loss_weight: float) -> torch.Tensor:
        self.data_idx = data_idx
        self.Tetra_coord=Tetra_coord
        self.Dir_Triangle_coord=Dir_Triangle_coord
        self.Pre_Triangle_coord=Pre_Triangle_coord
        self.Sym_Triangle_coord=Sym_Triangle_coord
        self.Pre_load=Pre_load
        integral=GaussIntegral()
        integral_strainenergy=integral.Integral3D(self.StrainEnergy, cfg.n_int3D, Tetra_coord)
        integral_externalwork=integral.Integral2D(self.ExternalWork, cfg.n_int2D, Pre_Triangle_coord)
        # integral_boundaryloss=integral.Integral2D(self.BoundaryLoss, 3, Dir_Triangle_coord)
        integral_boundaryloss=self.BoundaryLoss(Dir_Triangle_coord, Sym_Triangle_coord)

        energy_loss = integral_strainenergy - integral_externalwork
        loss = energy_loss + loss_weight*integral_boundaryloss
        
        # print("Internal Energy:", integral_strainenergy.item())
        # print("External Work:", integral_externalwork.item())
        # print("Boundary Loss:", integral_boundaryloss.item())
        return loss, energy_loss, loss_weight*integral_boundaryloss

    def GetU(self, xyz_field: torch.Tensor) -> torch.Tensor:
        u = self.model(xyz_field, self.data_idx)
        return u
    
    def StrainEnergy(self, xyz_field: torch.Tensor) -> torch.Tensor:
        E=cfg.E
        nu=cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        xyz_field.requires_grad = True  # 为了计算位移场的梯度，这里需要设置为True
        pred_u = self.GetU(xyz_field)

        duxdxyz = grad(pred_u[:, :, 0], xyz_field, torch.ones_like(pred_u[:, :, 0]), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(pred_u[:, :, 1], xyz_field, torch.ones_like(pred_u[:, :, 1]), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[:, :, 2], xyz_field, torch.ones_like(pred_u[:, :, 2]), create_graph=True, retain_graph=True)[0]
        # Fxx = duxdxyz[:, :, 0].unsqueeze(2) + 1
        # Fxy = duxdxyz[:, :, 1].unsqueeze(2) + 0
        # Fxz = duxdxyz[:, :, 2].unsqueeze(2) + 0
        # Fyx = duydxyz[:, :, 0].unsqueeze(2) + 0
        # Fyy = duydxyz[:, :, 1].unsqueeze(2) + 1
        # Fyz = duydxyz[:, :, 2].unsqueeze(2) + 0
        # Fzx = duzdxyz[:, :, 0].unsqueeze(2) + 0
        # Fzy = duzdxyz[:, :, 1].unsqueeze(2) + 0
        # Fzz = duzdxyz[:, :, 2].unsqueeze(2) + 1
        # detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        # trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2

        # strainenergy_tmp = 0.5 * lam * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 3)
        # strainenergy = strainenergy_tmp[:, :, 0]

        grad_u = torch.stack([duxdxyz, duydxyz, duzdxyz], dim=-2)  # [N,4,3,3]
    
        # 计算变形梯度张量 F = I + ∇u
        I = torch.eye(3, device=self.dev)  # [3,3]
        F = I + grad_u  # [N,3,3]

        # 计算变形梯度的行列式 J = det(F)
        J = torch.det(F).unsqueeze(-1) # 变形梯度行列式
        # J_safe=nn.functional.softplus(J) # 防止负体积单元

        I1=torch.sum(F**2, dim=[-2, -1]).unsqueeze(-1)

        EPS=1e-8
        strainenergy_tmp = 0.5 * lam * (torch.log(J + EPS) * torch.log(J + EPS)) - mu * torch.log(J + EPS) + 0.5 * mu * (I1 - 3)
        strainenergy = strainenergy_tmp[:, :, 0]
    
        return strainenergy
    
    def ExternalWork(self, pressure_field: torch.Tensor) -> torch.Tensor:
        # 计算压力边界法向量
        edge1 = self.Pre_Triangle_coord[:, 1] - self.Pre_Triangle_coord[:, 0]  # v0 -> v1
        edge2 = self.Pre_Triangle_coord[:, 2] - self.Pre_Triangle_coord[:, 0]  # v0 -> v2
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)
        # 确保法向量与半径方向一致
        # radial_yz = self.Pre_Triangle_coord[:, 0, 1:3] # x轴为旋转轴的半径方向
        # radial = torch.cat([torch.zeros_like(radial_yz[:, 0:1]),  radial_yz], dim=1)
        # radial_xy = self.Pre_Triangle_coord[:, 0, 0:2] # z轴为旋转轴的半径方向
        # radial=torch.cat([radial_xy, torch.zeros_like(radial_xy[:, 0:1])], dim=1)  # [N, 3]
        # dot = torch.sum(normals * radial, dim=1)  # 计算点积 [N]
        # normals_unity[dot > 0] *= -1  # 若点积为正，反转法向量方向，因为压力向量前面乘了负号

        #"""计算外力做功"""
        u_pred = self.GetU(pressure_field)
        #计算每个单元的高斯积分点的压力向量
        p= self.Pre_load*normals_unity.unsqueeze(1).expand(-1, u_pred.size(1), -1)

        external_work = torch.sum( -u_pred * p, dim=-1)
        return external_work
    
    def BoundaryLoss(self, dirichlet_field: torch.Tensor, symmetry_field: torch.Tensor) -> torch.Tensor:
        #"""计算边界条件损失函数"""
        # 1.计算dirichlet边界条件的损失函数
        u_dir_pred = self.GetU(dirichlet_field)

        u_dir_value=torch.tensor(cfg.Dir_u, dtype=torch.float32).to(self.dev)
        u_dir_true = torch.zeros_like(u_dir_pred)
        u_dir_true[:, :, :]=u_dir_value

        mes_loss = nn.MSELoss(reduction='sum')
        dir_loss = mes_loss(u_dir_pred, u_dir_true)
        
        # 2.计算对称边界条件的损失函数
        u_sym_pred = self.GetU(symmetry_field)
        ## 计算对称边界的法向量
        edge1 = symmetry_field[:, 1] - symmetry_field[:, 0]  # v0 -> v1
        edge2 = symmetry_field[:, 2] - symmetry_field[:, 0]  # v0 -> v2
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)
        normals_unity=normals_unity.unsqueeze(1).expand(-1, u_sym_pred.size(1), -1)
        ## 计算对称边界的法向位移
        u_sym_normal = torch.sum(u_sym_pred * normals_unity, dim=-1)
        u_sym_true = torch.zeros_like(u_sym_normal)
        sym_loss = mes_loss(u_sym_normal, u_sym_true)

        boundary_loss = dir_loss + sym_loss
        return boundary_loss

