import torch
import meshio
import time


class GaussIntegral:
    def __init__(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ctype = None
        self.n = None
        self.region = None

    def GaussPoints(self) -> torch.Tensor:
        if self.ctype == "tetra":
            if self.n == 1:
                xi_eta_zeta_weight = torch.tensor([
                    [0.25, 0.25, 0.25, 1.0]
                ])
            elif self.n == 2:
                xi_eta_zeta_weight = torch.tensor([
                    [0.1381966, 0.1381966, 0.1381966, 0.25],
                    [0.5854102, 0.1381966, 0.1381966, 0.25],
                    [0.1381966, 0.5854102, 0.1381966, 0.25],
                    [0.1381966, 0.1381966, 0.5854102, 0.25]
                ])
            elif self.n == 3:
                xi_eta_zeta_weight = torch.tensor([
                    [0.25, 0.25, 0.25, -0.8],
                    [0.5, 0.1666667, 0.1666667, 0.45],
                    [0.1666667, 0.5, 0.1666667, 0.45],
                    [0.1666667, 0.1666667, 0.5, 0.45],
                    [0.1666667, 0.1666667, 0.1666667, 0.45]
                ])
            elif self.n == 4:
                xi_eta_zeta_weight = torch.tensor([
                    [0.25, 0.25, 0.25, -0.078933333],
                    [0.07142857, 0.07142857, 0.07142857, 0.045733333],
                    [0.78571429, 0.07142857, 0.07142857, 0.045733333],
                    [0.07142857, 0.78571429, 0.07142857, 0.045733333],
                    [0.07142857, 0.07142857, 0.78571429, 0.045733333],
                    [0.39940358, 0.39940358, 0.10059642, 0.149333333],
                    [0.39940358, 0.10059642, 0.39940358, 0.149333333],
                    [0.10059642, 0.39940358, 0.39940358, 0.149333333],
                    [0.39940358, 0.10059642, 0.10059642, 0.149333333],
                    [0.10059642, 0.39940358, 0.10059642, 0.149333333],
                    [0.10059642, 0.10059642, 0.39940358, 0.149333333],
                ])
            else:
                raise ValueError("unsupported 3D Gauss order")

        if self.ctype == "triangle":
            if self.n == 1:
                xi_eta_zeta_weight = torch.tensor([
                    [1 / 3, 1 / 3, 1.0]
                ])
            elif self.n == 2:
                xi_eta_zeta_weight = torch.tensor([
                    [1 / 6, 1 / 6, 1 / 3],
                    [1 / 6, 2 / 3, 1 / 3],
                    [2 / 3, 1 / 6, 1 / 3]
                ])
            elif self.n == 3:
                xi_eta_zeta_weight = torch.tensor([
                    [1 / 3, 1 / 3, -27 / 48],
                    [3 / 5, 1 / 5, 25 / 48],
                    [1 / 5, 1 / 5, 25 / 48],
                    [1 / 5, 3 / 5, 25 / 48]
                ])
            else:
                raise ValueError("unsupported 2D Gauss order")

        return xi_eta_zeta_weight.to(self.dev)

    def ShapeFunctions(self, natural_coord: torch.Tensor) -> torch.Tensor:
        if self.ctype == "tetra":
            xi = natural_coord[:, 0]
            eta = natural_coord[:, 1]
            zeta = natural_coord[:, 2]
            n1 = 1 - xi - eta - zeta
            n2 = xi
            n3 = eta
            n4 = zeta
            n = torch.stack([n1, n2, n3, n4], dim=0)

        if self.ctype == "triangle":
            xi = natural_coord[:, 0]
            eta = natural_coord[:, 1]
            n1 = 1 - xi - eta
            n2 = xi
            n3 = eta
            n = torch.stack([n1, n2, n3], dim=0)

        return n

    def NaturalToPhysical(self, natural_coord: torch.Tensor) -> torch.Tensor:
        n = self.ShapeFunctions(natural_coord)
        x = torch.matmul(self.region[..., 0], n)
        y = torch.matmul(self.region[..., 1], n)
        z = torch.matmul(self.region[..., 2], n)
        return torch.stack([x, y, z], dim=-1)

    def JacobianDet(self):
        if self.ctype == "tetra":
            v1 = self.region[..., 1, :] - self.region[..., 0, :]
            v2 = self.region[..., 2, :] - self.region[..., 0, :]
            v3 = self.region[..., 3, :] - self.region[..., 0, :]
            matrix = torch.stack([v1, v2, v3], dim=-2)
            j_det = (1.0 / 6.0) * torch.abs(torch.linalg.det(matrix))
        if self.ctype == "triangle":
            v1 = self.region[..., 1, :] - self.region[..., 0, :]
            v2 = self.region[..., 2, :] - self.region[..., 0, :]
            v1v2 = torch.cross(v1, v2, dim=-1)
            j_det = 0.5 * torch.linalg.norm(v1v2, dim=-1)
        return j_det

    def Integral3D(self, f, n: int, region: torch.Tensor) -> torch.Tensor:
        self.ctype = 'tetra'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz = self.NaturalToPhysical(gauss_points)
        fv = f(xyz)
        weight = gauss_points[:, 3]
        j_det = self.JacobianDet()
        weighted = torch.sum(fv * weight, dim=-1)
        return torch.sum(weighted * j_det, dim=-1)

    def Integral2D(self, f, n: int, region: torch.Tensor) -> torch.Tensor:
        self.ctype = 'triangle'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz = self.NaturalToPhysical(gauss_points)
        fv = f(xyz)
        weight = gauss_points[:, 2]
        j_det = self.JacobianDet()
        weighted = torch.sum(fv * weight, dim=-1)
        return torch.sum(weighted * j_det, dim=-1)

    def Integral3D_batch(self, f, n: int, region: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        self.ctype = 'tetra'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz = self.NaturalToPhysical(gauss_points)
        fv = f(xyz)
        weight = gauss_points[:, 3]
        j_det = self.JacobianDet()
        weighted = torch.sum(fv * weight, dim=-1)
        return torch.sum(weighted * j_det * region_mask.float(), dim=-1)

    def Integral2D_batch(self, f, n: int, region: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        self.ctype = 'triangle'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz = self.NaturalToPhysical(gauss_points)
        fv = f(xyz)
        weight = gauss_points[:, 2]
        j_det = self.JacobianDet()
        weighted = torch.sum(fv * weight, dim=-1)
        return torch.sum(weighted * j_det * region_mask.float(), dim=-1)


if __name__ == '__main__':
    start_time = time.time()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mesh = meshio.read("DEFEM3D/Beam3D/beam_mesh.msh", file_format="gmsh")

    all_point_idx = mesh.cells_dict['tetra']
    tetra_coord = mesh.points[all_point_idx]
    tetra_coord = torch.tensor(tetra_coord, dtype=torch.float32).to(dev)

    neu_cell_idx = mesh.cell_sets_dict['bc_Neumann']['triangle']
    neu_point_idx = mesh.cells_dict['triangle'][neu_cell_idx]
    neu_coord = mesh.points[neu_point_idx]
    neu_coord = torch.tensor(neu_coord, dtype=torch.float32).to(dev)

    integral = GaussIntegral()

    def f(xyz):
        return xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2 + xyz[:, :, 2] ** 2

    integral_value = integral.Integral3D(f, 3, tetra_coord)
    integral_value_2d = integral.Integral2D(f, 3, neu_coord)

    end_time = time.time()
    print("time:", end_time - start_time, "s")
    print("vol:", integral_value)
    print("area:", integral_value_2d)
