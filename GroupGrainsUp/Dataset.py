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

    def _pad_tensor_dict(self, tensor_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(v.shape[0] for v in tensor_dict.values())
        sample = tensor_dict[0]
        padded = torch.zeros(
            (self.data_num, max_len, *sample.shape[1:]),
            dtype=sample.dtype,
            device=self.dev,
        )
        mask = torch.zeros((self.data_num, max_len), dtype=torch.bool, device=self.dev)
        for i in range(self.data_num):
            n = tensor_dict[i].shape[0]
            padded[i, :n] = tensor_dict[i]
            mask[i, :n] = True
        return padded, mask

    def _pad_tensor_dict_by_indices(self, tensor_dict: dict, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(tensor_dict[i].shape[0] for i in indices)
        sample = tensor_dict[indices[0]]
        padded = torch.zeros(
            (len(indices), max_len, *sample.shape[1:]),
            dtype=sample.dtype,
            device=self.dev,
        )
        mask = torch.zeros((len(indices), max_len), dtype=torch.bool, device=self.dev)
        for local_i, grain_i in enumerate(indices):
            n = tensor_dict[grain_i].shape[0]
            padded[local_i, :n] = tensor_dict[grain_i]
            mask[local_i, :n] = True
        return padded, mask

    def _build_batch_from_indices(self, indices: list[int], dom: dict, bc_dir: dict, bc_pre: dict, bc_sym: dict) -> dict:
        dom_pad, dom_mask = self._pad_tensor_dict_by_indices(dom, indices)
        dir_pad, dir_mask = self._pad_tensor_dict_by_indices(bc_dir, indices)
        pre_pad, pre_mask = self._pad_tensor_dict_by_indices(bc_pre, indices)
        sym_pad, sym_mask = self._pad_tensor_dict_by_indices(bc_sym, indices)
        return {
            "dom": dom_pad,
            "dom_mask": dom_mask,
            "bc_dir": dir_pad,
            "bc_dir_mask": dir_mask,
            "bc_pre": pre_pad,
            "bc_pre_mask": pre_mask,
            "bc_sym": sym_pad,
            "bc_sym_mask": sym_mask,
            "grain_idx": torch.tensor(indices, dtype=torch.long, device=self.dev),
            "grain_count": len(indices),
        }

    def batch_fields(self, Dir_marker: str, Pre_marker: str, Sym_marker: str) -> dict:
        dom = self.domain()
        bc_dir = self.bc_Dirichlet(Dir_marker)
        bc_pre = self.bc_Pressure(Pre_marker)
        bc_sym = self.bc_Symmetry(Sym_marker)

        dom_pad, dom_mask = self._pad_tensor_dict(dom)
        dir_pad, dir_mask = self._pad_tensor_dict(bc_dir)
        pre_pad, pre_mask = self._pad_tensor_dict(bc_pre)
        sym_pad, sym_mask = self._pad_tensor_dict(bc_sym)

        return {
            "dom": dom_pad,
            "dom_mask": dom_mask,
            "bc_dir": dir_pad,
            "bc_dir_mask": dir_mask,
            "bc_pre": pre_pad,
            "bc_pre_mask": pre_mask,
            "bc_sym": sym_pad,
            "bc_sym_mask": sym_mask,
            "grain_idx": torch.arange(self.data_num, dtype=torch.long, device=self.dev),
        }

    def batch_fields_bucketed(self, Dir_marker: str, Pre_marker: str, Sym_marker: str, bucket_num: int = 2) -> list[dict]:
        dom = self.domain()
        bc_dir = self.bc_Dirichlet(Dir_marker)
        bc_pre = self.bc_Pressure(Pre_marker)
        bc_sym = self.bc_Symmetry(Sym_marker)

        if bucket_num <= 1 or self.data_num <= 1:
            return [self._build_batch_from_indices(list(range(self.data_num)), dom, bc_dir, bc_pre, bc_sym)]

        # Sort by dominant tetrahedral element count to reduce padding waste in each bucket.
        sorted_indices = sorted(range(self.data_num), key=lambda i: dom[i].shape[0], reverse=True)
        bucket_num = min(bucket_num, self.data_num)
        step = (self.data_num + bucket_num - 1) // bucket_num

        buckets = []
        for start in range(0, self.data_num, step):
            idx_chunk = sorted_indices[start:start + step]
            buckets.append(self._build_batch_from_indices(idx_chunk, dom, bc_dir, bc_pre, bc_sym))

        return buckets

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
    
    
