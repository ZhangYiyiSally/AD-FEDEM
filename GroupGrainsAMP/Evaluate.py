import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.autograd import grad
from torch.autograd.functional import jacobian
import meshio
import numpy as np
import Config as cfg
from Dataset import Dataset
from Network import ResNet
from Loss import Loss
import Utility as util

# 选择GPU或CPU
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(model, xyz, idx, scaling):
    xyz=xyz*scaling
    xyz_tensor = torch.from_numpy(xyz).float()
    xyz_tensor = xyz_tensor.to(dev)
    xyz_tensor.requires_grad_(True)

    # 设置材料参数
    E=cfg.E # 杨氏模量
    nu =cfg.nu # 泊松比
    mu = E / (2 * (1 + nu)) # 剪切模量
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))   # 超弹性λ参数

    # 1. 通过神经网络预测位移u (形状: N, 3)
    u_pred= model(xyz_tensor, idx)

    # 2. 计算位移梯度 ∇u (形状: N, 3, 3)
    duxdxyz = grad(u_pred[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u_pred[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u_pred[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    grad_u = torch.stack([duxdxyz, duydxyz, duzdxyz], dim=1)  # [N,3,3]
    grad_norm = torch.norm(grad_u, dim=-1)  # [N,4,3]
    max_grad_norm = torch.max(grad_norm)
    print(f"最大梯度范数: {max_grad_norm.item()}")
    
    # 3. 计算变形梯度张量 F = I + ∇u
    I = torch.eye(3, device=dev).unsqueeze(0)  # [1,3,3]
    F = I + grad_u  # [N,3,3]

    # 4. 计算变形梯度的行列式 J = det(F), 并断言其大于0
    J = torch.det(F) # 变形梯度行列式
    assert torch.all(J > 1e-6), f"负体积单元: {torch.sum(J <= 0).item()}个" # 断言变形梯度行列式大于0

    # 5. 计算右Cauchy-Green张量 C = F^T F, 并计算其逆
    C = F.transpose(1,2) @ F # C = F^T F
    invC = torch.inverse(C) # C的逆
    
    # 6. 计算Green-Lagrange应变 E = 0.5*(C - I)
    epsilon = 0.5 * (C - I)
    principal_strains = torch.linalg.eigvalsh(epsilon)     # 形状 [n, 3]，升序排列
    e1 = principal_strains[:, -1]

    # 7. 计算第二类Piola-Kirchhoff应力S = mu*(I - C^(-1)) + λ*ln(J)*C^(-1)
    S = mu * (I - invC) + lam * torch.log(J).unsqueeze(-1).unsqueeze(-1) * invC

    # 8. 计算Cauchy应力sigma = (1/J) * F * S * F^T
    sigma = (1/J)[:, None, None] * torch.einsum('...ik,...kl,...lj->...ij', F, S, F.transpose(1,2))
    
    # 9. 计算Von Mises应力
    s11 = sigma[:, 0, 0]
    s22 = sigma[:, 1, 1]
    s33 = sigma[:, 2, 2]
    s12 = sigma[:, 0, 1]
    s23 = sigma[:, 1, 2]
    s13 = sigma[:, 2, 0]
    SVonMises = torch.sqrt(0.5 * (
        (s11 - s22)**2 + 
        (s22 - s33)**2 + 
        (s33 - s11)**2 + 
        6 * (s12**2 + s13**2 + s23**2)
    ))

    # 直接提取张量分量, 并转换为numpy格式便于写入vtu文件（保持索引一致性）
    # 预测的位移分量, tuple形式便于在vtu中计算合位移
    u = u_pred.detach().cpu().numpy()/scaling
    U = (np.float64(u[:,0]), np.float64(u[:,1]), np.float64(u[:,2]))
    # Green-Lagrange应变张量分量
    E11 = epsilon[:, 0, 0].detach().cpu().numpy()
    E12 = epsilon[:, 0, 1].detach().cpu().numpy()
    E13 = epsilon[:, 0, 2].detach().cpu().numpy()
    E22 = epsilon[:, 1, 1].detach().cpu().numpy()
    E23 = epsilon[:, 1, 2].detach().cpu().numpy()
    E33 = epsilon[:, 2, 2].detach().cpu().numpy()
    # 第二类Piola-Kirchhoff应力张量分量
    S11 = S[:, 0, 0].detach().cpu().numpy()
    S12 = S[:, 0, 1].detach().cpu().numpy()
    S13 = S[:, 0, 2].detach().cpu().numpy()
    S22 = S[:, 1, 1].detach().cpu().numpy()
    S23 = S[:, 1, 2].detach().cpu().numpy()
    S33 = S[:, 2, 2].detach().cpu().numpy()
    # Cauchy应力张量分量
    sigma11=s11.detach().cpu().numpy()
    sigma12=s12.detach().cpu().numpy()
    sigma13=s13.detach().cpu().numpy()
    sigma22=s22.detach().cpu().numpy()
    sigma23=s23.detach().cpu().numpy()
    sigma33=s33.detach().cpu().numpy()
    # Von Mises应力    
    SVonMises = SVonMises.detach().cpu().numpy()
    e1 = e1.detach().cpu().numpy()
    

    return U, SVonMises, e1, \
           E11, E12, E13, E22, E23, E33, \
           S11, S12, S13, S22, S23, S33, \
           sigma11, sigma12, sigma13, sigma22, sigma23, sigma33

# 从文件加载已经训练完成的模型
# model=MultiLayerNet(D_in=3, H=30, D_out=3).cuda()
dem_epoch=100000
model=ResNet(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, depth=cfg.depth, data_num=cfg.data_num, latent_dim=cfg.latent_dim).cuda()
model.load_state_dict(torch.load(f"{cfg.model_save_path}/dem_epoch{dem_epoch}.pth"))
model.eval()  # 设置模型为evaluation状态

# 读取有限元网格
# mesh0=meshio.read("DEFEM3D/Beam3D/beam_mesh0.msh", file_format="gmsh")
for data_idx in range(cfg.data_num):
    mesh = meshio.read(f'{cfg.mesh_path}/{data_idx}.msh', file_format="gmsh")

    # 计算该有限元网格对应的预测值
    xyz=mesh.points

    U, SVonMises, e1,\
    E11, E12, E13, E22, E23, E33, \
    S11, S12, S13, S22, S23, S33, \
    sigma11, sigma12, sigma13, sigma22, sigma23, sigma33= evaluate_model(model, xyz, data_idx, cfg.model_scale)

    # all_cells=mesh.cell_data['gmsh:physical'][6]
    # 写入vtu网格文件
    util.FEMmeshtoVTK(f"{cfg.Evaluate_save_path}/{data_idx}_epoch{dem_epoch}", mesh, pointdata={"U": U, "SVonMises": SVonMises, "e1":e1,\
                                                                                 "E11": E11, "E12": E12, "E13": E13, \
                                                                                 "E22": E22, "E23": E23, "E33": E33, \
                                                                                 "S11": S11, "S12": S12, "S13": S13, \
                                                                                 "S22": S22, "S23": S23, "S33": S33, \
                                                                                 "sigma11": sigma11, "sigma12": sigma12, "sigma13": sigma13, \
                                                                                 "sigma22": sigma22, "sigma23": sigma23, "sigma33": sigma33
                                                                                } )

    U_data=np.hstack((xyz, np.linalg.norm(np.array(U).T, axis=1).reshape(-1,1)))
    SVonMises_data=np.hstack((xyz, SVonMises.reshape(-1,1)))
    e1_data=np.hstack((xyz, e1.reshape(-1,1)))

    np.savetxt(f"{cfg.model_save_path}/{data_idx}_U_epoch{dem_epoch}.txt", U_data, fmt='%f', delimiter=' ')
    np.savetxt(f"{cfg.model_save_path}/{data_idx}_SVonMises_epoch{dem_epoch}.txt", SVonMises_data, fmt='%f', delimiter=' ')
    np.savetxt(f"{cfg.model_save_path}/e1_epoch{dem_epoch}.txt", e1_data, fmt='%f', delimiter=' ')
    print(f"模型dem_epoch{dem_epoch}.pth的预测结果已保存到{cfg.model_save_path}/{data_idx}_U_epoch{dem_epoch}.txt")
