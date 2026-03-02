import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import time
import meshio
from contextlib import nullcontext
import Config as cfg
from Network import ResNet
from Dataset import Dataset
from Loss import Loss

def plot_loss(loss, error):
    # 绘制损失曲线
    # plt.clf()
    # plt.figure(1)
    plt.plot(loss, linewidth=2, color='firebrick')
    plt.plot(error, linewidth=2, color='blue')
    plt.tick_params(axis='both', size=5, width=2,
                    direction='in', labelsize=15)
    plt.xlabel('epoch', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(['Loss', 'lr'], loc='upper right', fontsize=15)
    plt.title('Training Curve', size=20)
    plt.grid(color='midnightblue', linestyle='-.', linewidth=0.5)
    plt.ylim(-2000, 1)
    # 调整坐标轴的边框样式
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # 设置边框宽度为2
    plt.pause(0.0001)

def get_Pre_load(step:int, Pre_value:list, step_interval:int):
    initial_load = Pre_value[0] # 初始载荷
    max_load = Pre_value[1] # 最大载荷
    step_load = Pre_value[2] # 载荷步长
     # 计算应该增加多少次载荷
    load_increases = step // step_interval
    # 计算当前载荷
    current_load = initial_load + load_increases*step_load
    # 限制最大载荷
    pre_load = min(current_load, max_load)
    return pre_load

def get_loss_weight(step:int, loss_weight:list, step_interval:int):
    initial_weight = loss_weight[0] # 初始权重
    max_weight = loss_weight[1] # 最大权重
    step_weight = loss_weight[2] # 权重步长
    # 计算应该增加多少次权重
    weight_increases = step // step_interval
    # 计算当前权重
    current_weight = initial_weight + weight_increases*step_weight
    # 限制最大权重
    loss_weight = min(current_weight, max_weight)
    return loss_weight


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    torch.manual_seed(cfg.seed)

    data=Dataset(data_path=cfg.mesh_path, data_num=cfg.data_num, model_scale=cfg.model_scale)
    bucket_batches = data.batch_fields_bucketed(
        cfg.Dir_marker, cfg.Pre_marker, cfg.Sym_marker, getattr(cfg, "bucket_num", 2)
    )
    total_grain_count = sum(b["grain_count"] for b in bucket_batches)

    # 定义神经网络，神经网络的输入为空间坐标，输出为三个方向的位移
    dem=ResNet(cfg.input_size, cfg.hidden_size, cfg.output_size, cfg.depth, cfg.data_num, cfg.latent_dim).to(dev)
    start_epoch=0
    dem.train()
    loss = Loss(dem)

    # 开始训练, 设置训练参数
    start_time = time.time()
    losses = []
    epoch_num=cfg.epoch_num
    learning_rate=cfg.lr
        
    # 定义优化器
    # optimizer = torch.optim.LBFGS(dem.parameters(), lr=learning_rate_LBFGS, max_iter=max_iter_LBFGS)
    optimizer = torch.optim.Adam(dem.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(dem.parameters(), lr=learning_rate_SGD)

    # 定义学习率调度器
    lr_history = []
    amp_enabled = bool(getattr(cfg, "use_amp", False) and dev.type == "cuda")
    amp_dtype_name = getattr(cfg, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if cfg.lr_scheduler == 'Cos':
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)  # 每1000个epoch将学习率降低为原来的0.1倍
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)  # 指数衰减：每个epoch衰减为当前lr * gamma
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max,  eta_min=cfg.eta_min )  # 余弦退火：T_max为半周期（epoch数），eta_min为最小学习率
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=10, threshold=1e-4)  # 按指标衰减：当loss在patience个epoch内下降小于阈值threshold时，将学习率降低为原来的factor倍
        print(f"train: model_scale={cfg.model_scale}, Net={cfg.depth}x{cfg.input_size}-{cfg.hidden_size}-{cfg.output_size}({cfg.latent_dim}), lr={learning_rate:.0e}, scheduler={cfg.lr_scheduler}{scheduler.T_max}x{cfg.eta_min}, load={cfg.Pre_value[0]:.0e}, {cfg.Pre_value[2]:.0e}, {cfg.Pre_step_interval:.0e}, weight={cfg.loss_weight[0]:.0e}, {cfg.loss_weight[2]:.0e}, {cfg.weight_step_interval:.0e}")
    
    if cfg.lr_scheduler == 'Exp':
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)  # 每1000个epoch将学习率降低为原来的0.1倍
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)  # 指数衰减：每个epoch衰减为当前lr * gamma
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max,  eta_min=cfg.eta_min )  # 余弦退火：T_max为半周期（epoch数），eta_min为最小学习率
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=10, threshold=1e-4)  # 按指标衰减：当loss在patience个epoch内下降小于阈值threshold时，将学习率降低为原来的factor倍
        print(f"train: model_scale={cfg.model_scale}, Net={cfg.depth}x{cfg.input_size}-{cfg.hidden_size}-{cfg.output_size}({cfg.latent_dim}), lr={learning_rate:.0e}, scheduler={cfg.lr_scheduler}{scheduler.gamma}, load={cfg.Pre_value[0]:.0e}, {cfg.Pre_value[2]:.0e}, {cfg.Pre_step_interval:.0e}, weight={cfg.loss_weight[0]:.0e}, {cfg.loss_weight[2]:.0e}, {cfg.weight_step_interval:.0e}")

    tqdm_epoch = tqdm(range(start_epoch, epoch_num), desc='epoches',colour='red', dynamic_ncols=True)
    for epoch in range(start_epoch, epoch_num):
        Pre_load=get_Pre_load(epoch, cfg.Pre_value, cfg.Pre_step_interval)
        loss_weight=get_loss_weight(epoch, cfg.loss_weight, cfg.weight_step_interval)

        total_loss = torch.tensor(0.0, device=dev)
        total_eloss = torch.tensor(0.0, device=dev)
        total_bloss = torch.tensor(0.0, device=dev)
        optimizer.zero_grad()

        for batch in bucket_batches:
            bucket_weight = batch["grain_count"] / total_grain_count
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                bucket_loss, bucket_eloss, bucket_bloss = loss.loss_function_batch(
                    batch["grain_idx"],
                    batch["dom"], batch["dom_mask"],
                    batch["bc_dir"], batch["bc_dir_mask"],
                    batch["bc_pre"], batch["bc_pre_mask"],
                    batch["bc_sym"], batch["bc_sym_mask"],
                    Pre_load, loss_weight
                )
                weighted_loss = bucket_loss * bucket_weight
                total_loss = total_loss + weighted_loss.detach()
                total_eloss = total_eloss + (bucket_eloss * bucket_weight).detach()
                total_bloss = total_bloss + (bucket_bloss * bucket_weight).detach()

            if amp_enabled:
                scaler.scale(weighted_loss).backward()
            else:
                weighted_loss.backward()

        # 更新参数
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()  # 更新学习率

            # def closure():
            #     loss=Loss(dem)
            #     loss_closure=loss.loss_function(dom, bc_Dir, bc_Pre, bc_Sym)
            #     # 反向传播
            #     optimizer.zero_grad()
            #     loss_closure.backward()
            #     return loss_closure
            
            # optimizer.step(closure=closure)

        losses.append(float(total_loss.item()))
        lr_history.append(optimizer.param_groups[0]['lr'])


        # 更新epoch进度条
        tqdm_epoch.update()
        tqdm_epoch.set_postfix({'loss':'{:.5e}'.format(losses[-1]),'eloss':'{:.5e}'.format(total_eloss.item()), 'bloss':'{:.5e}'.format(total_bloss.item()), 'lr':'{:.1e}'.format(lr_history[-1]), 'p':'{:.1e}'.format(Pre_load), 'w':'{:.1e}'.format(loss_weight)})
        

        if epoch % 5000 == 0:
            # 保存模型
            os.makedirs(cfg.model_save_path, exist_ok=True)
            torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch}.pth")
            # plot_loss(losses, lr_history)
            # plt.savefig(f"{cfg.model_save_path}/training_curve_middle.png")
            with open(f"{cfg.model_save_path}/loss_middle_seed{cfg.seed}.txt", 'w') as f:
                f.write('\n'.join(map(str, losses)) + '\n')
    
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(dem.state_dict(), f"{cfg.model_save_path}/dem_epoch{epoch_num}.pth")
    plt.savefig(f"{cfg.model_save_path}/training_curve_epoch{epoch_num}.png")
    with open(f"{cfg.model_save_path}/loss_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, losses)) + '\n')
    # with open(f"{cfg.model_save_path}/errorL2_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        # f.write('\n'.join(map(str, eL2)) + '\n')
    # with open(f"{cfg.model_save_path}/errorH1_epoch{epoch_num}_seed{cfg.seed}.txt", 'w') as f:
        # f.write('\n'.join(map(str, eH1)) + '\n')
    with open(f"{cfg.model_save_path}/lr_epoch{epoch}_seed{cfg.seed}.txt", 'w') as f:
        f.write('\n'.join(map(str, lr_history)) + '\n')

    end_time=time.time()-start_time
    print("End time: %.5f" % end_time)
    print("训练结束：结果保存在%s" % cfg.model_save_path)
