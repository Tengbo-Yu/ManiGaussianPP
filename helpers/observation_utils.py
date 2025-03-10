import numpy as np
from rlbench.backend.observation import Observation

from rlbench.backend.observation import BimanualObservation
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode # 渲染模式
from typing import List

REMOVE_KEYS = [
    "joint_velocities",
    "joint_positions",
    "joint_forces",
    "gripper_open",
    "gripper_pose",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "task_low_dim_state",
    "misc",
]


def extract_obs(
    cfg,
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action=None,
    channels_last: bool = False,
    episode_length: int = 10,
    robot_name: str = "",
    next_obs: Observation = None # mani多的
):
    """在train时用的observation_utils.py 加载 data eval时用的是custom_rlbench_env.py"""
    #-------------------Mani--------------------
    # bimanual里面用了函数的方式，Mani里面只有一个函数
    #-------------------Mani--------------------
    if obs.is_bimanual:
        return extract_obs_bimanual(
            cfg,obs, cameras, t, prev_action, channels_last, episode_length, robot_name,
            next_obs# mani多的
        )
    else:
        return extract_obs_unimanual(
            cfg,obs, cameras, t, prev_action, channels_last, episode_length
        )


def extract_obs_unimanual(cfg,
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action=None,
    channels_last: bool = False,
    episode_length: int = 10,
):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None # mani多的
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(obs.gripper_joint_positions, 0.0, 0.04)
    # print("obs.right.gripper_joint_positions",obs.right.gripper_joint_positions)
    # print(" obs.gripper_joint_positions", obs.gripper_joint_positions)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = obs.get_low_dim_data()
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items() if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs.perception_data.items()
            if type(v) == np.ndarray or type(v) == list
        }
    else:
        # add extra dim to depth data
        obs_dict = {
            k: v if v.ndim == 3 else np.expand_dims(v, -1) for k, v in obs.perception_data.items()
        }
    obs_dict["low_dim_state"] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict["ignore_collisions"] = np.array([obs.ignore_collisions], dtype=np.float32)
    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict["%s_camera_extrinsics" % camera_name] = obs.misc["%s_camera_extrinsics" % camera_name]
        obs_dict["%s_camera_intrinsics" % camera_name] = obs.misc["%s_camera_intrinsics" % camera_name]
        # real 删掉
        # if not cfg.method.neural_renderer.use_nerf_picture:
        obs_dict["%s_next_camera_extrinsics" % camera_name] = obs.misc["%s_next_camera_extrinsics" % camera_name]
        obs_dict["%s_next_camera_intrinsics" % camera_name] = obs.misc["%s_next_camera_intrinsics" % camera_name]

    # add timestep to low_dim_state
    time = (1.0 - (t / float(episode_length - 1))) * 2.0 - 1.0
    obs_dict["low_dim_state"] = np.concatenate(
        [obs_dict["low_dim_state"], [time]]
    ).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict


def extract_obs_bimanual(cfg,
    obs: Observation,
    cameras,
    t: int = 0,
    prev_action = None,
    channels_last: bool = False,
    episode_length: int = 10,
    robot_name: str = "",
    next_obs: Observation = None, # mani多的
):
    # Mani区别多了right和left双臂
    obs.right.joint_velocities = None
    right_grip_mat = obs.right.gripper_matrix
    right_grip_pose = obs.right.gripper_pose
    right_joint_pos = obs.right.joint_positions
    obs.right.gripper_pose = None
    obs.right.gripper_matrix = None
    obs.right.joint_positions = None
    obs.right.wrist_camera_matrix = None    # mani多的

    obs.left.joint_velocities = None
    left_grip_mat = obs.left.gripper_matrix
    left_grip_pose = obs.left.gripper_pose
    left_joint_pos = obs.left.joint_positions
    obs.left.gripper_pose = None
    obs.left.gripper_matrix = None
    obs.left.joint_positions = None
    obs.left.wrist_camera_matrix = None    # mani多的

    # sim 真机 和仿真不一样
    if obs.right.gripper_joint_positions is not None:
        # 使用np.clip函数将右侧抓手关节位置限制在0.0到0.04的范围内
        obs.right.gripper_joint_positions = np.clip(
            obs.right.gripper_joint_positions, 0.0, 0.04
        )
        obs.left.gripper_joint_positions = np.clip(
            obs.left.gripper_joint_positions, 0.0, 0.04
        )
    # real
    # if obs.right.gripper_joint_positions is not None:
    #     obs.right.gripper_joint_positions = obs.right.gripper_joint_positions / 255
    #     obs.left.gripper_joint_positions = obs.left.gripper_joint_positions / 255  


        # print("obs_utils ## obs.right.gripper_joint_positions",obs.right.gripper_joint_positions) # [0.00443191 0.00415269](一堆输出)
        # print("obs_utils  ## obs.left.gripper_joint_positions",obs.left.gripper_joint_positions) # [0.0044179  0.00414202]

    
    # Mani增加---------------------------------------------
    # print("obs utils.py cfg.method.neural_renderer.use_nerf_picture = ",cfg.method.neural_renderer.use_nerf_picture)
    if cfg.method.neural_renderer.use_nerf_picture:
        if obs.nerf_multi_view_rgb is not None:
            nerf_multi_view_rgb = obs.nerf_multi_view_rgb
            # print("nerf_multi_view_rgb",nerf_multi_view_rgb)
        else:
            nerf_multi_view_rgb = None

        if obs.nerf_multi_view_depth is not None:
            nerf_multi_view_depth = obs.nerf_multi_view_depth
        else:
            nerf_multi_view_depth = None

        if obs.nerf_multi_view_camera is not None:
            nerf_multi_view_camera = obs.nerf_multi_view_camera
        else:
            nerf_multi_view_camera = None
    # Mani增加---------------------------------------------

    # fixme::
    # 使用 vars(obs) 将 obs 对象的属性转换为字典 obs_dict，字典的键是属性名，值是对应的属性值。
    obs_dict = vars(obs)
    # for k in obs_dict.keys():
    #     print("k = ",k)
    # for k in obs_dict["perception_data"].keys():
        # print("perception_key = ",k)
    # # for k in obs_dict["right"].keys():
    # print("perception_key",obs_dict["right"])       

    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    # print("front_mask",obs_dict["perception_data"]["front_mask"]) # 应该就是256*256个数字，代表标签
    # print(obs_dict["perception_data"]["front_mask"].shape) #(256,256)

    # 新的双臂 get_low_dim_data是啥!! ?? --[{2}]---------------------------------------------
    right_robot_state = obs.get_low_dim_data(obs.right)
    left_robot_state = obs.get_low_dim_data(obs.left)
    #原来Mani
    # robot_state = np.array([obs.gripper_open,*obs.gripper_joint_positions])

    # remove low-level proprioception variables that are not needed
    # 删除不需要的低水平本体感觉变量
    obs_dict = {k: v for k, v in obs_dict.items() if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        # 从最后一个 DIM 切换到第一个 DIM 的通道
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs.perception_data.items()
            if type(v) == np.ndarray or type(v) == list
        }
    else:
        # add extra dim to depth data 
        # 为深度数据添加额外的 DIM
        obs_dict = {
            k: v if v.ndim == 3 else np.expand_dims(v, -1) for k, v in obs.perception_data.items()
        }
    # for k in obs_dict.keys():
        # print("obs_dict = ",k) # 
    # for k in obs_dict["perception_data"].keys():
        # print("perception_key = ",k)
    # 双臂新增的 只看bimanual就行（单纯这个if）
    if robot_name == "right":
        obs_dict["low_dim_state"] = right_robot_state.astype(np.float32)
        # binary variable indicating if collisions are allowed or not while planning paths to reach poses
        obs_dict["ignore_collisions"] = np.array(
            [obs.right.ignore_collisions], dtype=np.float32
        )
    elif robot_name == "left":
        obs_dict["low_dim_state"] = left_robot_state.astype(np.float32)
        obs_dict["ignore_collisions"] = np.array(
            [obs.left.ignore_collisions], dtype=np.float32
        )
    elif robot_name == "bimanual":
        obs_dict["right_low_dim_state"] = right_robot_state.astype(np.float32)
        # -------------在这里已经错了--------------------------------------------------------------------------
        # print("obs_dict[right_low_dim_state]--------------------------------",obs_dict["right_low_dim_state"])
        # ---------------------------------------------------------------------------------------
        obs_dict["left_low_dim_state"] = left_robot_state.astype(np.float32)
        obs_dict["right_ignore_collisions"] = np.array(
            [obs.right.ignore_collisions], dtype=np.float32
        )
        obs_dict["left_ignore_collisions"] = np.array(
            [obs.left.ignore_collisions], dtype=np.float32
        )

    for k, v in [(k, v) for k, v in obs_dict.items() if "point_cloud" in k]:
        # ..TODO:: switch to np.float16
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict["%s_camera_extrinsics" % camera_name] = obs.misc["%s_camera_extrinsics" % camera_name]
        obs_dict["%s_camera_intrinsics" % camera_name] = obs.misc["%s_camera_intrinsics" % camera_name]
        # real的时候删掉
        # if not cfg.method.neural_renderer.use_nerf_picture:
        obs_dict["%s_next_camera_extrinsics" % camera_name] = obs.misc["%s_next_camera_extrinsics" % camera_name]
        obs_dict["%s_next_camera_intrinsics" % camera_name] = obs.misc["%s_next_camera_intrinsics" % camera_name]
        
    # add timestep to low_dim_state 
    # 将 TimeStep 添加到low_dim_state
    time = (1.0 - (t / float(episode_length - 1))) * 2.0 - 1.0

    # 双臂估计是else中运行的
    if "low_dim_state" in obs_dict:
        obs_dict["low_dim_state"] = np.concatenate(
            [obs_dict["low_dim_state"], [time]]
        ).astype(np.float32)
    else:
        obs_dict["right_low_dim_state"] = np.concatenate(
            [obs_dict["right_low_dim_state"], [time]]
        ).astype(np.float32)
        obs_dict["left_low_dim_state"] = np.concatenate(
            [obs_dict["left_low_dim_state"], [time]]
        ).astype(np.float32)

    obs.right.gripper_matrix = right_grip_mat
    obs.right.joint_positions = right_joint_pos
    obs.right.gripper_pose = right_grip_pose
    obs.left.gripper_matrix = left_grip_mat
    obs.left.joint_positions = left_joint_pos
    obs.left.gripper_pose = left_grip_pose
    # for key in right_grip_mat:
    #     print("helpers observation_utils.py --- right_grip_mat[key]", right_grip_mat[key])

    # Mani NERF 新增----------------------------------
    # print("obs utils.py cfg.method.neural_renderer.use_nerf_picture = ",cfg.method.neural_renderer.use_nerf_picture)
    if cfg.method.neural_renderer.use_nerf_picture:
        # if nerf_multi_view_rgb is not None:
        obs_dict['nerf_multi_view_rgb'] = nerf_multi_view_rgb
        # ---------在这里会输出一堆地址（21个一组）----------------------------------------------------------
        # print("helpers observation_utils.py --- obs_dict[nerf_multi_view_rgb]",obs_dict['nerf_multi_view_rgb'])
        # -------------------------------------------------------------------
        obs_dict['nerf_multi_view_depth'] = nerf_multi_view_depth
        obs_dict['nerf_multi_view_camera'] = nerf_multi_view_camera

        # for next frame prediction
        if next_obs is not None:
            if next_obs.nerf_multi_view_rgb is not None:
                obs_dict['nerf_next_multi_view_rgb'] = next_obs.nerf_multi_view_rgb
                obs_dict['nerf_next_multi_view_depth'] = next_obs.nerf_multi_view_depth
                obs_dict['nerf_next_multi_view_camera'] = next_obs.nerf_multi_view_camera
                # print("next_obs.nerf_multi_view_camera",next_obs.nerf_multi_view_camera)  # 说明有NERF参数
            else:
                obs_dict['nerf_next_multi_view_rgb'] = None
                obs_dict['nerf_next_multi_view_depth'] = None
                obs_dict['nerf_next_multi_view_camera'] = None
                # print("next_obs.nerf_multi_view_camera",next_obs.nerf_multi_view_camera)
            # 如果在这边加好像会出现最后一次还是第一次的错误
            # cameras = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"] 
            # for camera_name in cameras:
            #     if next_obs.perception_data[f"{camera_name}_mask"] is not None:
            #         obs_dict[f"{camera_name}_next_mask"] = next_obs.perception_data[f"{camera_name}_mask"]
            #     else:
            #         obs_dict[f"{camera_name}_next_mask"] = None
            #         # obs_dict["perception_data"][f"{camera_name}_next_mask"] = None

        # if next_obs is None, we do not add the next frame prediction
        # Mani NERF 新增----------------------------------

    return obs_dict


def create_obs_config(
    camera_names: List[str],
    camera_resolution: List[int],
    method_name: str,
    use_depth:bool = True,
    use_mask:bool = True,  # # real的时False候删掉 True, # 未改被调用时的传入！！
    robot_name: str = "bimanual",
    nerf_multi_view: bool = True,
):
    unused_cams = CameraConfig() # rlbench中
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=use_mask, #False,
        depth=use_depth, #mani 特有 元False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL,
        #nerf_multi_view =nerf_multi_view, # 为了同时跑peract新增
        #nerf_multi_view_mask =nerf_multi_view_mask,
    )

    # 键 是 camera_names 列表中的相机名称， 值 是 used_cams
    camera_configs = {camera_name: used_cams for camera_name in camera_names}
    # for keys in camera_configs:
        # print("camera_configs",keys) # over_shoulder_left等相机
    # 其中一些 obs 仅用于关键点检测。
    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        camera_configs=camera_configs,
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        robot_name=robot_name,
        nerf_multi_view=nerf_multi_view,
    )
    # for key in obs_config:
        # print("obs_config",key)
    # print(obs_config) # <rlbench.observation_config.ObservationConfig object at 0x72d49f1cb340>
    return obs_config
