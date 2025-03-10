# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}
#       bash train.sh PERACT_BC 0,1,2,3 12345 abc
#       bash scripts/train.sh PERACT_BC 0,1,2,3 12345 abcd
#       bash scripts/train.sh ManiGaussian_BC2 0,1,2,3 12345 mani
# 
# set the method name
method=${1} # PERACT_BC / BIMANUAL_PERACT  / ManiGaussian_BC2

# set the seed number
seed="0"
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True #True

# train_demo_path="/data1/zjyang/program/peract_bimanual/data2/train_data"
train_demo_path="/data1/zjyang/program/peract_bimanual/data_ntu/zips"
logdir="/data1/zjyang/program/peract_bimanual/log-mani/${exp_name}"
replay_path="/data1/zjyang/program/peract_bimanual/replay/real_ntu"
cameras=["front"]

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}
batch_size=1 # 1 #4 # 2

# tasks=[bimanual_pick_plate]
# tasks=[bimanual_pick_laptop,bimanual_straighten_rope,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
tasks=[handover_keyframe,lift_keyframe,pick_in_one_keyframe,pick_in_two_keyframe,press_keyframe]
num_view_for_nerf=1 #1 #21
use_dynamic_field=True # True #False
demo=30 #30 # 100
episode_length=10 #10 #25 # 20 # 4
save_freq=2500 # 2500
camera_resolution="[640,480]"
training_iterations=100001 # 100001
field_type='bimanual' # 'LF' # 'BIMANUAL' 'bimanual' 'LF'
mask_gen='bimanual' # 'nonerf' #'gt' # 'pre' 'None' # 这里必须改（neural判断少写了前面LF）
lambda_dyna=0.1 #0.1
render_freq=100 #2000
lambda_nerf=0.01 #1.0 # 0.01 # 0.01
mask_gt_rgb=True
warm_up=2000
mask_warm_up=3000 #0
lambda_embed=0.0
lambda_reg=0.0

lambda_dyna_leader=0.6  
lambda_mask=0.4            
lambda_mask_right=0.5 
lambda_next_loss_mask=0.5 # 无用

use_nerf_picture=True #False
# render输出尺寸
image_width=640 #256
image_height=480 #256


tmux select-pane -t 0 
# peract rlbench
tmux send-keys "conda activate rlbench; 
CUDA_VISIBLE_DEVICES=${train_gpu}  QT_AUTO_SCREEN_SCALE_FACTOR=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        rlbench.num_view_for_nerf=${num_view_for_nerf}\
        method.num_view_for_nerf=${num_view_for_nerf}\
        rlbench.cameras=${cameras}\
        method.neural_renderer.use_nerf_picture=${use_nerf_picture} \
        framework.save_freq=${save_freq} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        method.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        framework.training_iterations=${training_iterations} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        replay.path=${replay_path} \
        rlbench.episode_length=${episode_length} \
        rlbench.camera_resolution=${camera_resolution} \
        method.neural_renderer.lambda_nerf=${lambda_nerf} \
        method.neural_renderer.render_freq=${render_freq} \
        method.neural_renderer.image_width=${image_width} \
        method.neural_renderer.image_height=${image_height} \
        method.neural_renderer.lambda_embed=${lambda_embed} \
        method.neural_renderer.lambda_dyna=${lambda_dyna} \
        method.neural_renderer.lambda_reg=${lambda_reg} \
        method.neural_renderer.foundation_model_name=null \
        method.neural_renderer.use_dynamic_field=${use_dynamic_field} \
        method.neural_renderer.field_type=${field_type} \
        method.neural_renderer.mask_gen=${mask_gen} \
        method.neural_renderer.dataset.mask_gt_rgb=${mask_gt_rgb} \
        method.neural_renderer.lambda_dyna_leader=${lambda_dyna_leader} \
        method.neural_renderer.lambda_mask=${lambda_mask} \
        method.neural_renderer.lambda_mask_right=${lambda_mask_right} \
        method.neural_renderer.lambda_next_loss_mask=${lambda_next_loss_mask} \
        method.neural_renderer.next_mlp.warm_up=${warm_up} \
        method.neural_renderer.mask_warm_up=${mask_warm_up} 

"
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0
rm -rf log-mani/${exp_name}/${exp_name}/${method}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}