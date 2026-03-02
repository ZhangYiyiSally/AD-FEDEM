#---------------------------------随机种子-----------------------------------------
seed = 2025
#---------------------------------网格设置-----------------------------------------
data_num=2  # 训练数据集数量
model_scale=10 # 模型缩放比例
#--------------------------------高斯积分精度设置-----------------------------------------
n_int3D=2  # 三维高斯积分精度
n_int2D=2  # 二维高斯积分精度

#-------------------------------材料参数设置-------------------------------------
# ----------E：杨氏模量，nu：泊松比---------------
E=5e6  # 单位：Pa
nu=0.498  # 泊松比
#-------------------------------Dirichlet边界条件设置-------------------------------------
# ----------Dir_marker：边界标记，Dir_u：边界指定位移---------------
Dir_marker='OutSurface'
Dir_u=[0.0, 0.0, 0.0]
#-------------------------------压力边界条件设置-------------------------------------
# ----------Pre_marker：边界标记，Pre_value：边界指定力---------------
Pre_marker='InSurface'
Pre_value=[0.2e6, 7e6, 6.8e6]  # [初始值，最大值，步长]，单位：Pa
Pre_step_interval=50000  # 单位：步
#-------------------------------对称边界条件设置-------------------------------------
# ----------Sym_marker：边界标记---------------
Sym_marker='Symmetry'
#--------------------------------神经网络设置-----------------------------------------
input_size=3  # ResNet的输入大小
hidden_size=200  # ResNet的隐藏层大小
output_size=3  # ResNet的输出大小
depth=4  # ResNet的深度
latent_dim=256  # 隐变量的维度
#--------------------------------训练参数设置-----------------------------------------
epoch_num=100000  # 训练的epoch数
lr=4e-4 #  学习率
lr_scheduler='Exp'
gamma=0.9999
T_max=10000
eta_min=1e-6
loss_weight=[1e5, 1e8, 1e8-1e5] # 边界损失函数的权重 [初始值，最大值，步长]
weight_step_interval=50000  # 单位：步
parallel_reduce='mean'  # batch损失聚合方式: 'mean' or 'sum'

#---------------------------文件路径-----------------------------------------------
mesh_path=f"GroupGrainsUp/models"
if lr_scheduler == 'Cos':
    model_save_path=f"GroupGrainsUp/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{T_max}x{eta_min:.1e}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
    Evaluate_save_path=f"GroupGrainsUp/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{T_max}x{eta_min:.1e}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
if lr_scheduler == 'Exp':
    model_save_path=f"GroupGrainsUp/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{gamma}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
    Evaluate_save_path=f"GroupGrainsUp/Results/DataNum{data_num}/x{model_scale}_Net{depth}x{input_size}-{hidden_size}-{output_size}({latent_dim})_{lr_scheduler}{lr:.0e}_{gamma}/p[{Pre_value[0]/1e6}-{Pre_value[2]/1e6}-{Pre_step_interval:.0f}]xw[{loss_weight[0]:.0e}-{loss_weight[2]:.0e}-{weight_step_interval:.0f}]"
