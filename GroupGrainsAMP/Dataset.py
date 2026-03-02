import numpy as np
import torch
import meshio
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, data_path: str, data_num: int, model_scale: float):
        self.data_num = data_num
        self.model_scale = model_scale
        meshes = {}  # 存储所有网格数据
        for i in range(data_num):
            mesh = meshio.read(f"{data_path}/{i}.msh", file_format="gmsh")
            meshes[i] = mesh  # 将网格数据存储在字典中
            # meshes.append((mesh, i)) # 将网格和索引一起存储
        self.meshes = meshes
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def domain(self) -> torch.Tensor:
        Tetra_coord = {}
        for i in range(self.data_num):
            mesh = self.meshes[i]
            AllPoint_idx=mesh.cells_dict['tetra']
            Tetra_coord[i] = mesh.points[AllPoint_idx]
            Tetra_coord[i] = Tetra_coord[i]*self.model_scale # 缩放模型
            Tetra_coord[i] = torch.tensor(Tetra_coord[i], dtype=torch.float32).to(self.dev)

        return Tetra_coord
        
    def bc_Dirichlet(self, marker:str) -> torch.Tensor:
        Dir_Triangle_coord = {}
        for i in range(self.data_num):
            mesh = self.meshes[i]
            DirCell_idx=mesh.cell_sets_dict[marker]['triangle']
            DirPoint_idx=mesh.cells_dict['triangle'][DirCell_idx]
            Dir_Triangle_coord[i] = mesh.points[DirPoint_idx]
            Dir_Triangle_coord[i] = Dir_Triangle_coord[i]*self.model_scale # 缩放模型
            Dir_Triangle_coord[i] = torch.tensor(Dir_Triangle_coord[i], dtype=torch.float32).to(self.dev)
            
        return Dir_Triangle_coord
    
    def bc_Pressure(self, marker:str) -> torch.Tensor:
        Pre_Triangle_coord = {}
        for i in range(self.data_num):
            mesh = self.meshes[i]
            PreCell_idx=mesh.cell_sets_dict[marker]['triangle']
            PrePoint_idx=mesh.cells_dict['triangle'][PreCell_idx]
            Pre_Triangle_coord[i] = mesh.points[PrePoint_idx]
            Pre_Triangle_coord[i] = Pre_Triangle_coord[i]*self.model_scale # 缩放模型
            Pre_Triangle_coord[i] = torch.tensor(Pre_Triangle_coord[i], dtype=torch.float32).to(self.dev)

        return Pre_Triangle_coord
    
    def bc_Symmetry(self, marker:str) -> torch.Tensor:
        Sym_Triangle_coord = {}
        for i in range(self.data_num):
            mesh = self.meshes[i]
            SymCell_idx=mesh.cell_sets_dict[marker]['triangle']
            SymPoint_idx=mesh.cells_dict['triangle'][SymCell_idx]
            Sym_Triangle_coord[i] = mesh.points[SymPoint_idx]
            Sym_Triangle_coord[i] = Sym_Triangle_coord[i]*self.model_scale # 缩放模型
            Sym_Triangle_coord[i] = torch.tensor(Sym_Triangle_coord[i], dtype=torch.float32).to(self.dev)

        return Sym_Triangle_coord
    
if __name__ == '__main__':  # 测试边界条件是否设置正确
    data = Dataset(data_path='DEFEM3D/GroupGrains/models', data_num=2)
    dom = data.domain()
    Dir_coord = data.bc_Dirichlet('OutSurface')
    Pre_coord = data.bc_Pressure('InSurface')
    Sym_coord = data.bc_Symmetry('Symmetry')

    print("全域四面体单元个数*单元顶点个数*坐标方向:", dom[0].shape, dom[1].shape)
    print("Dirichlet边界三角形单元个数*单元顶点个数*坐标方向:", Dir_coord[0].shape, Dir_coord[1].shape)
    print("Pressure 边界三角形单元个数*单元顶点个数*坐标方向:", Pre_coord[0].shape, Pre_coord[1].shape)
    print("Symmetry边界三角形单元个数*单元顶点个数*坐标方向:", Sym_coord[0].shape, Sym_coord[1].shape)
    
    