#!/bin/bash


# 定义需要测试的文件夹路径数组: 1. dpo; 2. grpo-dispered
folders=(
    # "/database/wuyonghuang/hsam_code/output/sam2/20250307_150328_Synapse_512_pretrain_vit_l_epo300_bs8_lr0.0001_s2345"
    "/database/wuyonghuang/hsam_code/output/sam2/20250307_170124_Synapse_512_pretrain_vit_l_epo300_bs8_lr0.0001_s2345"
)


# 遍历文件夹数组
for folder in "${folders[@]}"; do
    # 使用 find 命令查找所有 .pth 文件
    pth_files=$(find "$folder" -type f -name "*.pth" | sort)
    
    # 将找到的 .pth 文件传递给 Python 脚本
    while IFS= read -r pth_file; do
		current_time=$(date +"%Y-%m-%d-%H:%M:%S")

		CUDA_VISIBLE_DEVICES=$1 python test.py \
		--stage=3 --img_size=512 --model=sam2 --vit_name="vit_l" \
		--is_savenii \
		--ckpt=$pth_file \
		--prompt_type='point'  \
		> /database/wuyonghuang/hsam_code/test_outputs/${current_time}.log

    done <<< "$pth_files"
done
