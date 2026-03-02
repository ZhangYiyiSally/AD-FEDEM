import gmsh
import Config as cfg

# n=20
# for i in range(n):
#     # 初始化 Gmsh
#     gmsh.initialize()
#     # 加载 .geo 文件
#     gmsh.open(f"{cfg.mesh_path}/{i}.geo")
#     print(f"运行文件：{cfg.mesh_path}/{i}.geo")
gmsh.initialize()
gmsh.open(f"{cfg.mesh_path}/11.geo")