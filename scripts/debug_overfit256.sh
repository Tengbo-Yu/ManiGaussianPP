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
# use_wandb=True
use_wandb=True
train_demo_path="/mnt/disk_1/tengbo/bimanual_data/train"
# replay_path="/data1/zjyang/program/peract_bimanual/replay/debug/overfit/multiepisode"
# replay_path="/data1/zjyang/program/peract_bimanual/replay/debug/overfit/duoepisode"
replay_path="/mnt/disk_1/tengbo/ManiGaussian++/replay"
# replay_path="/mnt/disk_2/tengbo/replay"
# replay_path="/data1/zjyang/program/peract_bimanual/replay_debug/"

addition_info="$(date +%Y%m%d)"
# exp_name=${4:- "debug"}
exp_name=${4:-"${method}_${addition_info}"}
# logdir="/data1/zjyang/program/peract_bimanual/log-mani/${exp_name}"
logdir="/mnt/disk_1/tengbo/ManiGaussian++/log/${exp_name}"
# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}
batch_size=1 # 1 #4 # 2

# tasks=[dual_push_buttons] bimanual_pick_plate
# tasks=[bimanual_sweep_to_dustpan] # dual_push_buttons bimanual_straighten_rope 
# bimanual_pick_laptop coordinated_lift_tray coordinated_push_box 
# handover_item_easy coordinated_take_tray_out_of_oven coordinated_put_bottle_in_fridge
# handover_item bimanual_sweep_to_dustpan
tasks=[bimanual_straighten_rope]
# tasks=[bimanual_pick_laptop,bimanual_straighten_rope,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]

num_view_for_nerf=21 #1 #21
# for debug
use_dynamic_field=False #True # False # True #False
apply_se3=False
render_freq=100 # 100 #2000

use_neural_rendering=True #False
demo=1 # 100
episode_length=25 # 1 #2 #25 # 20 # 4
save_freq=10000
camera_resolution="[256,256]"
training_iterations=100001
field_type='LF' # 'LF' # 'BIMANUAL' 'bimanual' 'LF'
lambda_dyna=0.1 #0.1
lambda_reg=0.0
lambda_nerf=1.0 # 0.01 # 0.01
mask_gt_rgb=True
lambda_dyna_leader=0.6  
lambda_mask=0.0 #0.4            
lambda_mask_right=0.4 
lambda_next_loss_mask=0.5

# action loss
lambda_bc=0.0

warm_up=1000 #400 #0
mask_warm_up=3000 #0

lambda_embed=0.0
mask_gen='pre' #'bimanual' # 'pre' # 'nonerf' #'gt' # 'pre' 'None'
use_nerf_picture=True
image_width=256
image_height=256

tmux select-pane -t 0 
# peract rlbench
tmux send-keys "conda activate mg; 
CUDA_VISIBLE_DEVICES=${train_gpu}  QT_AUTO_SCREEN_SCALE_FACTOR=0 TORCH_DISTRIBUTED_DEBUG=DETAIL python train.py method=$method \
        method.lambda_bc=${lambda_bc} \
        method.transform_augmentation.apply_se3=${apply_se3} \
        method.use_neural_rendering=${use_neural_rendering} \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        rlbench.num_view_for_nerf=${num_view_for_nerf}\
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
        method.neural_renderer.foundation_model_name=null\
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
# remove 0.ckpt diffusion
# rm -rf logs/${exp_name}/seed${seed}/weights/0
rm -rf log-mani/${exp_name}/${exp_name}/${method}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}