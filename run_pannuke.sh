cd /database/wuyonghuang/hsam_code
source /database/wuyonghuang/hsam_code/uv_hsam/bin/activate


for dpo_weight in 0.0001 0	0.1 0.05 0.01 0.001  #   # 或者简写为 seq start end，如果 increment 为 1
do
  echo "当前数字: $dpo_weight"
  current_time=$(date +"%Y-%m-%d-%H:%M:%S")
  CUDA_VISIBLE_DEVICES=$1  python train.py \
	--dataset='PanNuke' \
	--batch_size=8 --base_lr=0.0005 --weight_decay=0.01 --kl_beta=0.05 \
	--img_size=224 \
	--warmup --AdamW --max_epochs=50 \
	--vit_name='vit_b' \
	--n_gpu 1 \
	--model 'sam2' \
	--ours_use_lora \
	--is_dpo --dpo_weight=$dpo_weight \
	--prompt_type='point' \
	--pos_point_num="(1, 5)" \
	--neg_point_num="(0, 3)" \
	--kl_prompt_type='point' \
	--interval_epoch=1 \
	--num_prompts_per_class=3 \
	--exp_series="lora_dpo_weight_pannuke_exp" \
	--exp_name="exp1" \
	> /database/wuyonghuang/hsam_code/output/lora_DPO-dpo_weight_pannuke_${current_time}-${dpo_weight}.log 2>&1 \
	|| python -c "from sende import let_me_know; let_me_know('lora_DPO_weight_exp_${dpo_weight}_Error', 'SAM-Exps')"
done