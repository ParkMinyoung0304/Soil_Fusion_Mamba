import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil

from tensorboardX import SummaryWriter
from utils.split_dataset_cutting import split_dataset

# 设置路径
sub_img_folder = r'/home/czh/Sigma-main/datasets/Soil/Cutting'
train_txt_path = r'/home/czh/Sigma-main/datasets/Soil/train.txt'
test_txt_path = r'/home/czh/Sigma-main/datasets/Soil/test.txt'
extra_save_dir = r'/home/czh/Sigma-main/models/encoders/selective_scan/datasets/Soil'

# 调用函数分割数据集
split_dataset(sub_img_folder, train_txt_path, test_txt_path, extra_save_dir)

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '16005'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    print(args)
    
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        from configs.config_MFNet import config
    elif dataset_name == 'pst':
        from configs.config_pst900 import config
    elif dataset_name == 'nyu':
        from configs.config_nyu import config
    elif dataset_name == 'sun':
        from configs.config_sunrgbd import config
    elif dataset_name == 'soil':
        from configs.config_Soil import config
    else:
        raise ValueError('Not a valid dataset name')

    print(config.tb_dir)

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    # 在这里插入标签分布检查代码
    # print("正在检查训练集标签分布...")
    # label_counts = {i: 0 for i in range(256)}  # 初始化所有可能的标签值
    # total_pixels = 0

    # for batch_idx, batch in enumerate(tqdm(train_loader, desc="处理批次")):
    #     if batch_idx == 0:  # 只打印第一个批次的信息
    #         print("\nbatch 的结构:")
    #         for key, value in batch.items():
    #             if isinstance(value, torch.Tensor):
    #                 print(f"  {key}: 形状 {value.shape}, 类型 {value.dtype}")
    #             else:
    #                 print(f"  {key}: 类型 {type(value)}")
            
    #         print("\n'label' 张量的一些统计信息:")
    #         labels = batch['label']
    #         print(f"  形状: {labels.shape}")
    #         print(f"  数据类型: {labels.dtype}")
    #         print(f"  最小值: {labels.min().item()}")
    #         print(f"  最大值: {labels.max().item()}")
    #         print(f"  唯一值: {torch.unique(labels).tolist()}")
            
    #         print("\n第一张图像的标签的前10x10像素:")
    #         print(labels[0, :10, :10])
            
    #         # 如果batch中包含图像数据,也可以打印图像的一些信息
    #         if 'data' in batch:
    #             print("\n'data' 张量的一些统计信息:")
    #             data = batch['data']
    #             print(f"  形状: {data.shape}")
    #             print(f"  数据类型: {data.dtype}")
    #             print(f"  最小值: {data.min().item()}")
    #             print(f"  最大值: {data.max().item()}")
        
    #     labels = batch['label']
    #     if torch.any(labels == 255):
    #         print(f"\n在批次 {batch_idx} 中发现255标签")
    #         print(f"255标签的位置: {(labels == 255).nonzero()}")
    #         print(f"255标签的数量: {(labels == 255).sum().item()}")
            
    #         # 检查255是否主要出现在边缘
    #         edge_mask = torch.zeros_like(labels, dtype=torch.bool)
    #         edge_mask[:, 0, :] = edge_mask[:, -1, :] = edge_mask[:, :, 0] = edge_mask[:, :, -1] = True
    #         edge_255 = ((labels == 255) & edge_mask).sum().item()
    #         total_255 = (labels == 255).sum().item()
    #         print(f"边缘255标签占比: {edge_255 / total_255 * 100:.2f}%")

    #         if batch_idx > 5:  # 只检查前几个批次
    #             break

    #     # 处理批次的其余部分...
    #     unique_values, counts = torch.unique(labels, return_counts=True)
    #     for value, count in zip(unique_values.tolist(), counts.tolist()):
    #         label_counts[value] += count
    #         total_pixels += count

    # print("\n标签分布统计：")
    # for index, count in sorted(label_counts.items()):
    #     if count > 0:
    #         percentage = (count / total_pixels) * 100
    #         print(f"  标签 {index}: {count} 像素 ({percentage:.2f}%)")

    # print(f"\n总像素数：{total_pixels}")

    # if 255 in label_counts:
    #     background_percentage = (label_counts[255] / total_pixels) * 100
    #     print(f"背景像素 (255) 占比: {background_percentage:.2f}%")

    # # 检查是否有超出预期范围的标签
    # unexpected_labels = [label for label in label_counts if label_counts[label] > 0 and label not in range(config.num_classes) and label != 255]
    # if unexpected_labels:
    #     print("\n警告：发现意外的标签值：")
    #     for label in unexpected_labels:
    #         print(f"  标签 {label}: {label_counts[label]} 像素")



    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    # Initialize the evaluation dataset and evaluator
    val_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = ValPre()
    val_dataset = RGBXDataset(val_setting, 'val', val_pre)

    best_mean_iou = 0.0  # Track the best mean IoU for model saving
    best_epoch = 100000  # Track the epoch with the best mean IoU for model saving
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
                # 在这里添加打印语句
            # print("标签值t的统计信息:")
            # print(f"最小值: {gts.min().item()}")
            # print(f"最大值: {gts.max().item()}")
            # print(f"唯一值: {torch.unique(gts).tolist()}")
            # print(f"形状: {gts.shape}")
            # print(f"数据类型: {gts.dtype}")
            # print(f"类别数量: {config.num_classes}")

            # if gts.numel() <= 100:
            #     print(f"所有标签值: {gts.tolist()}")
            # else:
            #     print(f"前100个标签值: {gts.flatten()[:100].tolist()}")

            # invalid_labels = (gts < 0) | (gts >= config.num_classes)
            # if invalid_labels.any():
            #     print("警告: 发现无效的标签值!")
            #     print(f"无效标签的位置: {torch.nonzero(invalid_labels)}")
            #     print(f"无效标签的值: {gts[invalid_labels]}")

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                pbar.set_description(print_str, refresh=False)
            del loss
            
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
        
        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()
        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        all_dev = parse_devices(args.devices)
                        # network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d).cuda(all_dev[0])
                        segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                network=model, multi_scales=config.eval_scale_array,
                                                is_flip=config.eval_flip, devices=[model.device],
                                                verbose=False, config=config,
                                                )
                        print(f"Checkpoint directory: {config.checkpoint_dir}")
                        print(f"Checkpoint directory exists: {os.path.exists(config.checkpoint_dir)}")
                        print(f"Epoch: {epoch}")
                        print(f"Val log file: {config.val_log_file}")
                        print(f"Link val log file: {config.link_val_log_file}")

                        _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                    config.link_val_log_file)
                        print('mean_IoU:', mean_IoU)
                        
                        # Determine if the model performance improved
                        if mean_IoU > best_mean_iou:
                            # If the model improves, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                        else:
                            # If the model does not improve, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                        
                    model.train()
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval() 
                with torch.no_grad():
                    devices_val = [engine.local_rank] if engine.distributed else [0]
                    segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                            norm_mean=config.norm_mean, norm_std=config.norm_std,
                                            network=model, multi_scales=config.eval_scale_array,
                                            is_flip=config.eval_flip, devices=[0],
                                            verbose=False, config=config,
                                            )
                    _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                config.link_val_log_file)
                    print('mean_IoU:', mean_IoU)
                    
                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()
