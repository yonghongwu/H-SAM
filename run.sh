current_time=$(date +"%Y-%m-%d-%H:%M:%S")

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

# note: DPO 是可以运行的
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
CUDA_VISIBLE_DEVICES=$1  python train.py \
    --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
	--split='train' \
	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
	--img_size=512 \
	--warmup --AdamW --max_epochs=300 --stop_epoch=300 \
	--vit_name='vit_l' \
	--n_gpu 1 \
	--model 'sam2' \
	--is_grpo --rw_dispered \
	--interval_epoch=1 \
	--num_prompts_per_class=3 \
	> /database/wuyonghuang/hsam_code/output/${current_time}.log 2>&1 \
	|| python -c "from sende import let_me_know; let_me_know('Error', 'SAM-Exps')"