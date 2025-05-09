# note: 已经完成实验并统计; 实验结果请在飞书文档中查看; 2 4 8 非常差
# 当前的 命令中, 由于新加了 onlybest_in_multimask_output 参数, 因此在没有store_true 的情况下, 默认就会在sam的多输出中随机选择一个 预测
# for dpo_weight in 0.01 0.001 0.0001  # 或者简写为 seq start end，如果 increment 为 1
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

# 更大的size
# for dpo_weight in 0 0.001 0.0001  # 或者简写为 seq start end，如果 increment 为 1
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
# 	--vit_name='vit_l' \
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


# 使用的是 vanilla 版本: 不设置 --is_grpo --is_dpo --dev 的情况下, 是使用vanilla
for iter_name in 0
do
  current_time=$(date +"%Y-%m-%d-%H:%M:%S")
  CUDA_VISIBLE_DEVICES=$1  python train.py \
    --root_path="/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512" \
	--split='train' \
	--batch_size=8 --base_lr=0.0001 --weight_decay=0.01 --kl_beta=0.05 \
	--img_size=224 \
	--warmup --AdamW --max_epochs=5 \
	--vit_name='vit_b' \
	--n_gpu 1 \
	--model 'sam2' \
	--prompt_type='point' \
	--pos_point_num="(1, 5)" \
	--neg_point_num="(0, 3)" \
	--kl_prompt_type='both' \
	--interval_epoch=1 \
	--num_prompts_per_class=3 \
	--exp_series="dev_exp" \
	--exp_name="exp4-onlybest_in_multimask_output" \
	> /database/wuyonghuang/hsam_code/output/logs/dev_${current_time}.log 2>&1 \
	|| python -c "from sende import let_me_know; let_me_know('dev_Error', 'SAM-Exps')"
done