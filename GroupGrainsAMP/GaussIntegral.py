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
        # """计算高斯积分点和权重"""
        if self.ctype=="tetra":
        # 四面体的高斯积分点和权重
            if self.n==1:
             xi_eta_zeta_weight= torch.tensor([
                [0.25, 0.25, 0.25, 1.0]  # 高斯点1
                ])
            elif self.n==2:
                xi_eta_zeta_weight= torch.tensor([
                [0.1381966, 0.1381966, 0.1381966, 0.25],  # 高斯点1
                [0.5854102, 0.1381966, 0.1381966, 0.25],  # 高斯点2
                [0.1381966, 0.5854102, 0.1381966, 0.25],  # 高斯点3
                [0.1381966, 0.1381966, 0.5854102, 0.25]   # 高斯点4
                ])
            elif self.n==3:
                xi_eta_zeta_weight= torch.tensor([
                [0.25, 0.25, 0.25, -0.8],  # 高斯点1
                [0.5, 0.1666667, 0.1666667, 0.45],  # 高斯点2
                [0.1666667, 0.5, 0.1666667, 0.45],  # 高斯点3
                [0.1666667, 0.1666667, 0.5, 0.45],  # 高斯点4
                [0.1666667, 0.1666667, 0.1666667, 0.45]   # 高斯点5
                ])
            elif self.n==4:
                xi_eta_zeta_weight= torch.tensor([
                [0.25, 0.25, 0.25, -0.078933333],  # 高斯点1
                [0.07142857,0.07142857,0.07142857, 0.045733333],  # 高斯点2
                [0.78571429,0.07142857,0.07142857, 0.045733333],  # 高斯点3
                [0.07142857,0.78571429,0.07142857, 0.045733333],  # 高斯点4
                [0.07142857,0.07142857,0.78571429, 0.045733333],   # 高斯点5
                [0.39940358,0.39940358,0.10059642, 0.149333333],  # 高斯点6
                [0.39940358,0.10059642,0.39940358, 0.149333333],  # 高斯点7
                [0.10059642,0.39940358,0.39940358, 0.149333333],  # 高斯点8
                [0.39940358,0.10059642,0.10059642, 0.149333333],   # 高斯点9
                [0.10059642,0.39940358,0.10059642, 0.149333333],   # 高斯点10
                [0.10059642,0.10059642,0.39940358, 0.149333333],   # 高斯点11
                ])
            else:
                raise ValueError("3D高斯积分阶数不支持")

        if self.ctype=="triangle":
        # 三角形的高斯积分点和权重
            if self.n==1:
                xi_eta_zeta_weight= torch.tensor([
                [1/3, 1/3, 1.0]  # 高斯点1
                ])
            elif self.n==2:
                xi_eta_zeta_weight= torch.tensor([
                [1/6, 1/6, 1/3],  # 高斯点1
                [1/6, 2/3, 1/3],  # 高斯点2
                [2/3, 1/6, 1/3]   # 高斯点3
                ])
            elif self.n==3:
                xi_eta_zeta_weight= torch.tensor([
                [1/3, 1/3, -27/48],  # 高斯点1
                [3/5, 1/5, 25/48],  # 高斯点2
                [1/5, 1/5, 25/48],  # 高斯点3
                [1/5, 3/5, 25/48]   # 高斯点4
                ])
            else:
                raise ValueError("2D高斯积分阶数不支持")
        return xi_eta_zeta_weight.to(self.dev)

    def ShapeFunctions(self, natural_coord: torch.Tensor) -> torch.Tensor:
        # """计算形函数"""
        if self.ctype=="tetra":
            # """计算四面体的形函数"""
            xi=natural_coord[:, 0]
            eta=natural_coord[:, 1]
            zeta=natural_coord[:, 2]
            N1 = 1 - xi - eta - zeta
            N2 = xi
            N3 = eta
            N4 = zeta
            N = torch.stack([N1, N2, N3, N4], dim=0)

        if self.ctype=="triangle":
            # """计算三角形的形函数"""
            xi=natural_coord[:, 0]
            eta=natural_coord[:, 1]
            N1 = 1 - xi - eta
            N2 = xi
            N3 = eta
            N = torch.stack([N1, N2, N3], dim=0)

        return N 
    
    def NaturalToPhysical(self, natural_coord: torch.Tensor) -> torch.Tensor:
        # """将自然坐标转换为物理坐标"""
        if self.ctype=="tetra":
            # """四面体单元"""
            N = self.ShapeFunctions(natural_coord)
            x = torch.matmul(self.region[:, :, 0], N )
            y = torch.matmul(self.region[:, :, 1], N ) 
            z = torch.matmul(self.region[:, :, 2], N )
            physical_coord = torch.stack([x, y, z], dim=0).permute(1, 2, 0)
        if self.ctype=="triangle":
            # """三角形单元"""
            N = self.ShapeFunctions(natural_coord)
            x = torch.matmul(self.region[:, :, 0], N )  
            y = torch.matmul(self.region[:, :, 1], N )
            z = torch.matmul(self.region[:, :, 2], N ) 
            physical_coord = torch.stack([x, y, z], dim=0).permute(1, 2, 0)

        return physical_coord
        
    def JacobianDet(self):
        #"""计算雅可比矩阵的归一化行列式"""
        if self.ctype=="tetra":
            #"""四面体单元计算体积"""
            v1=self.region[:, 1] - self.region[:, 0]
            v2=self.region[:, 2] - self.region[:, 0]
            v3=self.region[:, 3] - self.region[:, 0]
            matrix_tmp=torch.stack([v1, v2, v3], dim=0)
            matrix=matrix_tmp.permute(1, 0, 2)
            J_det = 1/6 * torch.abs(torch.linalg.det(matrix))
        if self.ctype=="triangle":
            #"""三角形单元"""
            v1=self.region[:, 1] - self.region[:, 0]
            v2=self.region[:, 2] - self.region[:, 0]
            v1v2=torch.cross(v1, v2, dim=-1)
            J_det = 1/2* torch.linalg.norm(v1v2, axis=1)
        return J_det
    
    def Integral3D(self, f, n: int, region: torch.Tensor) -> torch.Tensor:
        #"""计算三维体积分值"""
        self.ctype = 'tetra'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz=self.NaturalToPhysical(gauss_points)
        f=f(xyz)
        weight=gauss_points[:,3]
        J_det=self.JacobianDet()
        integral_value = torch.matmul(torch.matmul(f, weight), J_det)
        return integral_value
    
    def Integral2D(self, f, n: int, region: torch.Tensor) -> torch.Tensor:
        #"""计算二维面积分值"""
        self.ctype = 'triangle'
        self.n = n
        self.region = region
        gauss_points = self.GaussPoints()
        xyz=self.NaturalToPhysical(gauss_points)
        f=f(xyz)
        weight=gauss_points[:,2]
        J_det=self.JacobianDet()
        integral_value = torch.matmul(torch.matmul(f, weight), J_det)
        return integral_value

if __name__ == '__main__':
    start_time=time.time()
    dev=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mesh=meshio.read("DEFEM3D/Beam3D/beam_mesh.msh", file_format="gmsh")

    # 提取四面体单元顶点坐标数组：m*4*3（m为四面体单元个数，4为四面体的四个顶点，3为三维空间坐标）
    AllPoint_idx=mesh.cells_dict['tetra']
    Tetra_coord=mesh.points[AllPoint_idx]
    Tetra_coord=torch.tensor(Tetra_coord, dtype=torch.float32)
    Tetra_coord=Tetra_coord.to(dev)

    # 提取纽曼边界条件的三角形单元顶点坐标数组：m*3*3（m为三角形单元个数，3为三角形的三个顶点，3为三维空间坐标）
    NeuCell_idx=mesh.cell_sets_dict['bc_Neumann']['triangle']
    NeuPoint_idx=mesh.cells_dict['triangle'][NeuCell_idx]
    Neu_coord = mesh.points[NeuPoint_idx]
    Neu_coord=torch.tensor(Neu_coord, dtype=torch.float32)
    Neu_coord=Neu_coord.to(dev)

    # 计算积分值
    integral=GaussIntegral()
    def f(xyz):
        return xyz[:, :, 0]**2 + xyz[:, :, 1]**2 + xyz[:, :, 2]**2
    integral_value=integral.Integral3D(f, 3, Tetra_coord)
    integral_value_2D=integral.Integral2D(f, 3, Neu_coord)

    end_time=time.time()
    print("计算时间:", end_time-start_time, "s")
    print("体积分值:", integral_value)
    print("面积分值:", integral_value_2D)