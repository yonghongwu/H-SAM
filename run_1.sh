cd /database/wuyonghuang/hsam_code
source /database/wuyonghuang/hsam_code/uv_hsam/bin/activate
# failed: 
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#        	--root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=2 --base_lr=0.0001 --weight_decay=0.01 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--interval_epoch=1 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# doing: KL和奖励都用了, 增加梯度裁剪: 还是会出现 NaN, 甚至还提前出现了，本来是1229, 变成 800了
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_po \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# # fail: KL和奖励都用了, 增加梯度裁剪, 提高 beta_kl=1,也提前了，从800到690
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=1 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_po \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# doing: KL和奖励都用了, 增加梯度裁剪, 设置 beta_kl=0, 900的时候出现
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_po \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# doing: KL和奖励都用了, 增加梯度裁剪, 设置 beta_kl=0.05, gt 作为掩码，只计算内部的损失
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_grpo \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"

# DPO 是可以运行的
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# fail: 将GRPO的奖励修改成离散
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
# 	--vit_name='vit_l' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_grpo --rw_dispered \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# 增加 采样数量为 8
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=4 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=3 --stop_epoch=3 \
# 	--vit_name='vit_b' \
# 	--n_gpu=1 \
# 	--model='sam2' \
# 	--is_grpo --rw_dispered \
# 	--rw_temp=3 \
# 	--grpo_KL_weight \
# 	--weight_temp=0.5 \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=8 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"

# 增加采样数量为16
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=2 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=3 --stop_epoch=3 \
# 	--vit_name='vit_b' \
# 	--n_gpu=1 \
# 	--model='sam2' \
# 	--is_grpo --rw_dispered \
# 	--rw_temp=3 \
# 	--grpo_KL_weight \
# 	--weight_temp=0.5 \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=16 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# 增加采样数量为8, 使用 f1 奖励函数, KL 权重设置为 1
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=4 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=1 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=3 --stop_epoch=3 \
# 	--vit_name='vit_b' \
# 	--n_gpu=1 \
# 	--model='sam2' \
# 	--is_grpo --rw_dispered --rw_func='f1' \
# 	--rw_temp=3 \
# 	--grpo_KL_weight \
# 	--weight_temp=0.5 \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=8 \
# 	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"

# # dev
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=4 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=1 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=3 --stop_epoch=3 \
# 	--vit_name='vit_b' \
# 	--n_gpu=1 \
# 	--model='sam2' \
# 	--dev \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=8 \
# 	> /database/wuyonghuang/hsam_code/output/dev-${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# # DPO: 跑了 50 个epoch, DPO + kl
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=50 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/DPO-${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"

# # DPO: 跑了 50 个epoch, 只有 DPO
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=50 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo --abla_kl \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	--exp_series="dpo_ablaKL_exp" \
# 	> /database/wuyonghuang/hsam_code/output/DPO_ablaKL-${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# # DPO: 跑了 50 个epoch, 只有 kl, 比 DPO 好太多, 前者甚至不 work
# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=50 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo --abla_dpo \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/DPO-${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"


# CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=50 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo --dpo_weight=0.1 \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	> /database/wuyonghuang/hsam_code/output/DPO-${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('little_DPO_weight_exp_Error', 'SAM-Exps')"


# note: 已经完成实验并统计; 实验结果请在飞书文档中查看; 1 3 5 10 非常差
# 当前的 命令中, 由于新加了 onlybest_in_multimask_output 参数, 因此在没有store_true 的情况下, 默认就会在sam的多输出中随机选择一个 预测
# for dpo_weight in 0	0.1 0.05 # 0.01 0.001 0.0001  # 或者简写为 seq start end，如果 increment 为 1
# do
#   # 在这里执行你的操作，可以使用 $i 引用当前数字
#   echo "当前数字: $dpo_weight"
#   current_time=$(date +"%Y-%m-%d-%H:%M:%S")
#   CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=5 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo --dpo_weight=$dpo_weight \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	--exp_series="dpo_weight_exp" \
# 	--exp_name="exp4-onlybest_in_multimask_output" \
# 	> /database/wuyonghuang/hsam_code/output/DPO-dpo_weight_${current_time}-${dpo_weight}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('little_DPO_weight_exp_${dpo_weight}_Error', 'SAM-Exps')"
# done

# note: 更大的 vit size; 已经完成实验并统计; 实验结果请在飞书文档中查看
# for dpo_weight in 0.1 0.05 0.01 # 0.001 0.0001  # 或者简写为 seq start end，如果 increment 为 1
# do
#   # 在这里执行你的操作，可以使用 $i 引用当前数字
#   echo "当前数字: $dpo_weight"
#   current_time=$(date +"%Y-%m-%d-%H:%M:%S")
#   CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=512 \
# 	--warmup --AdamW --max_epochs=5 \
# 	--vit_name='vit_' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--is_dpo --dpo_weight=$dpo_weight \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	--exp_series="dpo_weight_exp" \
# 	> /database/wuyonghuang/hsam_code/output/DPO-dpo_weight_${current_time}-${dpo_weight}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('little_DPO_weight_exp_${dpo_weight}_Error', 'SAM-Exps')"
# done


# # Bug: 使用 dev 版本, 但是这个版本会出现报错, 现在不调试了
# for iter_name in 0
# do
#   current_time=$(date +"%Y-%m-%d-%H:%M:%S")
#   CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=5 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model 'sam2' \
# 	--dev \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=3 \
# 	--exp_series="dev_exp" \
# 	--exp_name="exp4-onlybest_in_multimask_output" \
# 	> /database/wuyonghuang/hsam_code/output/logs/dev_${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('dev_Error', 'SAM-Exps')"
# done


# 再尝试grpo, 不设置 grpo_KL_weight=True 和 --weight_temp=0.5 和 rw_dispered=True # Bug 不想尝试了
# for exp_idx in 1
# do
#   # 在这里执行你的操作，可以使用 $i 引用当前数字
#   current_time=$(date +"%Y-%m-%d-%H:%M:%S")
#   CUDA_VISIBLE_DEVICES=$1  python train.py \
#     --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
# 	--split='train' \
# 	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
# 	--img_size=224 \
# 	--warmup --AdamW --max_epochs=5 \
# 	--vit_name='vit_b' \
# 	--n_gpu 1 \
# 	--model='sam2' \
#     --is_grpo --rw_func='f1' --rw_temp=3  \
# 	--prompt_type='point' \
# 	--pos_point_num="(1, 5)" \
# 	--neg_point_num="(0, 3)" \
# 	--kl_prompt_type='both' \
# 	--interval_epoch=1 \
# 	--num_prompts_per_class=6 \
# 	--exp_series="grpo_exp" \
#     --exp_name="尝试grpo-v5" \
# 	> /database/wuyonghuang/hsam_code/output/grpo_exp1_${current_time}.log 2>&1 \
# 	|| python -c "from sende import let_me_know; let_me_know('grpo_v5_Error', 'SAM-Exps')"
# done


for exp_idx in 1
do
  # 在这里执行你的操作，可以使用 $i 引用当前数字
  current_time=$(date +"%Y-%m-%d-%H:%M:%S")
  CUDA_VISIBLE_DEVICES=$1  python train.py \
    --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
	--split='train_clean' \
	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
	--img_size=224 \
	--warmup --AdamW --max_epochs=5 \
	--n_gpu 1 \
	--interval_epoch=1 \
	--exp_series="train_unet" \
    --exp_name="v1" \
	--only_train_unet
	> /database/wuyonghuang/hsam_code/output/unet_v1_${current_time}.log 2>&1 \
	|| python -c "from sende import let_me_know; let_me_know('unet_v1_Error', 'SAM-Exps')"
done