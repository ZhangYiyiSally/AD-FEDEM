from torch.autograd import grad
import torch
import torch.nn as nn
import Config as cfg
from GaussIntegral import GaussIntegral


class Loss:
    def __init__(self, model):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model

    def loss_function(self, data_idx, Tetra_coord: torch.Tensor, Dir_Triangle_coord: torch.Tensor, Pre_Triangle_coord: torch.Tensor, Sym_Triangle_coord: torch.Tensor, Pre_load: float, loss_weight: float) -> torch.Tensor:
        self.data_idx = data_idx
        self.Tetra_coord = Tetra_coord
        self.Dir_Triangle_coord = Dir_Triangle_coord
        self.Pre_Triangle_coord = Pre_Triangle_coord
        self.Sym_Triangle_coord = Sym_Triangle_coord
        self.Pre_load = Pre_load

        integral = GaussIntegral()
        integral_strainenergy = integral.Integral3D(self.StrainEnergy, cfg.n_int3D, Tetra_coord)
        integral_externalwork = integral.Integral2D(self.ExternalWork, cfg.n_int2D, Pre_Triangle_coord)
        integral_boundaryloss = self.BoundaryLoss(Dir_Triangle_coord, Sym_Triangle_coord)

        energy_loss = integral_strainenergy - integral_externalwork
        loss = energy_loss + loss_weight * integral_boundaryloss

        return loss, energy_loss, loss_weight * integral_boundaryloss

    def loss_function_batch(
        self,
        grain_idx: torch.Tensor,
        Tetra_coord: torch.Tensor,
        Tetra_mask: torch.Tensor,
        Dir_coord: torch.Tensor,
        Dir_mask: torch.Tensor,
        Pre_coord: torch.Tensor,
        Pre_mask: torch.Tensor,
        Sym_coord: torch.Tensor,
        Sym_mask: torch.Tensor,
        Pre_load: float,
        loss_weight: float,
    ):
        self.data_idx_batch = grain_idx
        self.Pre_Triangle_coord_batch = Pre_coord
        self.Pre_mask_batch = Pre_mask

        integral = GaussIntegral()
        strain_energy = integral.Integral3D_batch(self.StrainEnergy_batch, cfg.n_int3D, Tetra_coord, Tetra_mask)
        external_work = integral.Integral2D_batch(self.ExternalWork_batch, cfg.n_int2D, Pre_coord, Pre_mask)
        boundary_loss = self.BoundaryLoss_batch(Dir_coord, Dir_mask, Sym_coord, Sym_mask)

        energy_loss = strain_energy - external_work
        loss_per_grain = energy_loss + loss_weight * boundary_loss

        reduce_mode = getattr(cfg, "parallel_reduce", "mean")
        if reduce_mode == "sum":
            total_loss = torch.sum(loss_per_grain)
            total_energy = torch.sum(energy_loss)
            total_boundary = torch.sum(loss_weight * boundary_loss)
        else:
            total_loss = torch.mean(loss_per_grain)
            total_energy = torch.mean(energy_loss)
            total_boundary = torch.mean(loss_weight * boundary_loss)

        return total_loss, total_energy, total_boundary

    def GetU(self, xyz_field: torch.Tensor) -> torch.Tensor:
        return self.model(xyz_field, self.data_idx)

    def GetU_batch(self, xyz_field: torch.Tensor) -> torch.Tensor:
        return self.model(xyz_field, self.data_idx_batch)

    def StrainEnergy(self, xyz_field: torch.Tensor) -> torch.Tensor:
        E = cfg.E
        nu = cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        xyz_field.requires_grad = True
        pred_u = self.GetU(xyz_field)

        duxdxyz = grad(pred_u[:, :, 0], xyz_field, torch.ones_like(pred_u[:, :, 0]), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(pred_u[:, :, 1], xyz_field, torch.ones_like(pred_u[:, :, 1]), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[:, :, 2], xyz_field, torch.ones_like(pred_u[:, :, 2]), create_graph=True, retain_graph=True)[0]

        grad_u = torch.stack([duxdxyz, duydxyz, duzdxyz], dim=-2)

        I = torch.eye(3, device=self.dev)
        F = I + grad_u

        J = torch.det(F).unsqueeze(-1)
        I1 = torch.sum(F ** 2, dim=[-2, -1]).unsqueeze(-1)

        eps = 1e-8
        strainenergy_tmp = 0.5 * lam * (torch.log(J + eps) * torch.log(J + eps)) - mu * torch.log(J + eps) + 0.5 * mu * (I1 - 3)
        strainenergy = strainenergy_tmp[:, :, 0]

        return strainenergy

    def StrainEnergy_batch(self, xyz_field: torch.Tensor) -> torch.Tensor:
        E = cfg.E
        nu = cfg.nu
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        xyz_field = xyz_field.requires_grad_(True)
        pred_u = self.GetU_batch(xyz_field)

        duxdxyz = grad(pred_u[..., 0], xyz_field, torch.ones_like(pred_u[..., 0]), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(pred_u[..., 1], xyz_field, torch.ones_like(pred_u[..., 1]), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(pred_u[..., 2], xyz_field, torch.ones_like(pred_u[..., 2]), create_graph=True, retain_graph=True)[0]

        grad_u = torch.stack([duxdxyz, duydxyz, duzdxyz], dim=-2)

        I = torch.eye(3, device=self.dev)
        F = I + grad_u

        J = torch.det(F).unsqueeze(-1)
        I1 = torch.sum(F ** 2, dim=[-2, -1]).unsqueeze(-1)

        eps = 1e-8
        strainenergy_tmp = 0.5 * lam * (torch.log(J + eps) * torch.log(J + eps)) - mu * torch.log(J + eps) + 0.5 * mu * (I1 - 3)
        return strainenergy_tmp[..., 0]

    def ExternalWork(self, pressure_field: torch.Tensor) -> torch.Tensor:
        edge1 = self.Pre_Triangle_coord[:, 1] - self.Pre_Triangle_coord[:, 0]
        edge2 = self.Pre_Triangle_coord[:, 2] - self.Pre_Triangle_coord[:, 0]
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)

        u_pred = self.GetU(pressure_field)
        p = self.Pre_load * normals_unity.unsqueeze(1).expand(-1, u_pred.size(1), -1)

        external_work = torch.sum(-u_pred * p, dim=-1)
        return external_work

    def ExternalWork_batch(self, pressure_field: torch.Tensor) -> torch.Tensor:
        edge1 = self.Pre_Triangle_coord_batch[..., 1, :] - self.Pre_Triangle_coord_batch[..., 0, :]
        edge2 = self.Pre_Triangle_coord_batch[..., 2, :] - self.Pre_Triangle_coord_batch[..., 0, :]
        normals = torch.cross(edge1, edge2, dim=-1)
        norm = torch.norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
        normals_unity = normals / norm

        u_pred = self.GetU_batch(pressure_field)
        normals_expanded = normals_unity.unsqueeze(-2).expand(-1, -1, u_pred.size(-2), -1)
        p = self.Pre_load * normals_expanded

        external_work = torch.sum(-u_pred * p, dim=-1)
        return external_work * self.Pre_mask_batch.float().unsqueeze(-1)

    def BoundaryLoss(self, dirichlet_field: torch.Tensor, symmetry_field: torch.Tensor) -> torch.Tensor:
        u_dir_pred = self.GetU(dirichlet_field)

        u_dir_value = torch.tensor(cfg.Dir_u, dtype=torch.float32).to(self.dev)
        u_dir_true = torch.zeros_like(u_dir_pred)
        u_dir_true[:, :, :] = u_dir_value

        mes_loss = nn.MSELoss(reduction='sum')
        dir_loss = mes_loss(u_dir_pred, u_dir_true)

        u_sym_pred = self.GetU(symmetry_field)
        edge1 = symmetry_field[:, 1] - symmetry_field[:, 0]
        edge2 = symmetry_field[:, 2] - symmetry_field[:, 0]
        normals = torch.cross(edge1, edge2, dim=1)
        normals_unity = normals / torch.norm(normals, dim=1, keepdim=True)
        normals_unity = normals_unity.unsqueeze(1).expand(-1, u_sym_pred.size(1), -1)

        u_sym_normal = torch.sum(u_sym_pred * normals_unity, dim=-1)
        u_sym_true = torch.zeros_like(u_sym_normal)
        sym_loss = mes_loss(u_sym_normal, u_sym_true)

        boundary_loss = dir_loss + sym_loss
        return boundary_loss

    def BoundaryLoss_batch(self, dirichlet_field: torch.Tensor, dir_mask: torch.Tensor, symmetry_field: torch.Tensor, sym_mask: torch.Tensor) -> torch.Tensor:
        u_dir_pred = self.GetU_batch(dirichlet_field)
        u_dir_value = torch.tensor(cfg.Dir_u, dtype=torch.float32, device=self.dev)
        u_dir_true = torch.zeros_like(u_dir_pred)
        u_dir_true[..., :] = u_dir_value
        dir_sq = (u_dir_pred - u_dir_true).pow(2).sum(dim=(-1, -2))
        dir_loss = torch.sum(dir_sq * dir_mask.float(), dim=-1)

        u_sym_pred = self.GetU_batch(symmetry_field)
        edge1 = symmetry_field[..., 1, :] - symmetry_field[..., 0, :]
        edge2 = symmetry_field[..., 2, :] - symmetry_field[..., 0, :]
        normals = torch.cross(edge1, edge2, dim=-1)
        norm = torch.norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
        normals_unity = normals / norm
        normals_expanded = normals_unity.unsqueeze(-2).expand(-1, -1, u_sym_pred.size(-2), -1)

        u_sym_normal = torch.sum(u_sym_pred * normals_expanded, dim=-1)
        sym_sq = u_sym_normal.pow(2).sum(dim=-1)
        sym_loss = torch.sum(sym_sq * sym_mask.float(), dim=-1)

        return dir_loss + sym_loss
