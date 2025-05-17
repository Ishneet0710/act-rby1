import time
import numpy as np # type: ignore
import mujoco # type: ignore
import mujoco.viewer # type: ignore
import h5py # type: ignore
import os

def initialize_hdf5_storage(data_dir="data", episode_idx=1):
    """
    Prepares storage for one episode:
      - Creates `data/episode_<n>` directory
      - Initializes Python lists for qpos, ctrl, and each camera stream.
    """
    # make sure the directory exists
    episode_dir = os.path.join(data_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)                                        

    # storage buffers
    storage = {
        "qpos": [],
        "qvel": [], 
        "ctrl": [],
        "images": {
            "front_camera": [],
            "cam_ee_l": [],
            "cam_ee_r": []
        },
        "episode_dir": episode_dir
    }
    return storage

def store_data(storage, data, renderer):
    """
    Appends one frame of data to storage:
      - `data.qpos` (joint positions)
      - `data.ctrl` (motor commands)
      - RGB images from the three onboard cameras
    """
    # joint state and control
    storage["qpos"].append(data.qpos.copy())  
    storage["qvel"].append(data.qvel.copy())                                     
    storage["ctrl"].append(data.ctrl.copy())                                       

    # helper to capture and store images
    def capture(cam_name):
        renderer.update_scene(data, camera=cam_name)                                
        img = renderer.render()                                                      
        storage["images"][cam_name].append(img)

    # grab from each camera
    for cam in storage["images"]:
        capture(cam)



def final_save(storage):
    """
    Writes accumulated lists into an HDF5 file:
      - Datasets: /observations/qpos, /observations/ctrl
      - Image groups: /observations/images/<cam_name>
    """
    path = os.path.join(storage["episode_dir"], "recording.hdf5")
    with h5py.File(path, "w") as f:                                                
        obs = f.create_group("observations")                                       

        # write numeric arrays
        obs.create_dataset("qpos",   data=np.array(storage["qpos"]),   compression="gzip")  
        obs.create_dataset("qvel",   data=np.array(storage["qvel"]),   compression="gzip")  # ← save velocities
        obs.create_dataset("ctrl",   data=np.array(storage["ctrl"]),   compression="gzip")  

        # write images under a subgroup
        img_grp = obs.create_group("images")
        for cam_name, imgs in storage["images"].items():
            img_grp.create_dataset(
                cam_name,
                data=np.array(imgs),
                compression="gzip",
                chunks=(1, *imgs[0].shape)                                       
            )
    print(f"Saved recording to {path}")

DT             = 0.002
INTEGRATION_DT = 0.1
DAMPING        = 1e-4
KPOS           = 0.95
KORI           = 0.95
MAX_ANGVEL     = 0.785
NULLSPACE_GAIN = 1.0
POS_THRESHOLD  = 0.1

def set_gripper_position_left(model, data, open_fraction):
    left_act_id = model.actuator("left_finger_act").id
    desired_position = open_fraction * 0.05
    data.ctrl[left_act_id] = desired_position

def set_gripper_position_right(model, data, open_fraction):
    right_act_id = model.actuator("right_finger_act").id
    desired_position = open_fraction * 0.05
    data.ctrl[right_act_id] = desired_position

def is_finger_near_target(model, data, finger_joint_name, open_fraction, tol=0.02):
    finger_qpos_addr = model.joint(finger_joint_name).qposadr[0]
    desired_angle = open_fraction * 0.05
    current_angle = data.qpos[finger_qpos_addr]
    return abs(current_angle - desired_angle) < tol

def compute_dq(model, data, dof_ids, home_qpos, site_name, target_pos, target_quat):
    site_id = model.site(site_name).id
    current_pos = data.site_xpos[site_id]
    mat_3x3 = data.site_xmat[site_id].reshape((3, 3))
    cur_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cur_quat, mat_3x3.ravel())

    # Position error
    pos_err = target_pos - current_pos
    twist = np.zeros(6)
    twist[:3] = KPOS * pos_err / INTEGRATION_DT

    # Orientation error (quaternion-based)
    conj_cur = np.array([cur_quat[0], -cur_quat[1], -cur_quat[2], -cur_quat[3]])
    err_quat = np.zeros(4)
    mujoco.mju_mulQuat(err_quat, target_quat, conj_cur)
    ang_vel = np.zeros(3)
    mujoco.mju_quat2Vel(ang_vel, err_quat, 1.0)
    twist[3:] = KORI * ang_vel / INTEGRATION_DT

    # Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    full_jac = np.vstack((jacp, jacr))
    J = full_jac[:, dof_ids]

    # Damped least-squares
    damp_eye = DAMPING * np.eye(6)
    inv_term = np.linalg.inv(J @ J.T + damp_eye)
    dq_raw = J.T @ (inv_term @ twist)

    # Nullspace term
    q_now = data.qpos[dof_ids]
    diff_home = home_qpos - q_now
    J_pinv = np.linalg.pinv(J)
    I = np.eye(len(dof_ids))
    null_term = (I - J_pinv @ J) @ (NULLSPACE_GAIN * diff_home)

    dq = dq_raw + null_term

    # Clamp joint velocities
    max_val = np.max(np.abs(dq))
    if max_val > MAX_ANGVEL:
        dq *= (MAX_ANGVEL / max_val)

    return dq

def main():
    ABOVE      = 0
    DESCEND    = 1
    CLOSE_GRIP = 2
    LIFT       = 3
    DONE       = 4
    state = ABOVE

    model_path = "scene_arm.xml"  
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    torso_left_joints = [
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
        "left_arm_4", "left_arm_5", "left_arm_6"
    ]
    dof_ids_left = [model.joint(name).qposadr[0] for name in torso_left_joints]

    actuator_names_left = [
        "link1_act", "link2_act", "link3_act", "link4_act", "link5_act", "link6_act",
        "left_arm_1_act", "left_arm_2_act", "left_arm_3_act",
        "left_arm_4_act", "left_arm_5_act", "left_arm_6_act", "left_arm_7_act"
    ]
    actuator_ids_left = [model.actuator(a_name).id for a_name in actuator_names_left]
    left_finger_joint = "gripper_finger_l2"
    ee_site_name_left = "ee_site_l"

    torso_right_joints = [
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
        "right_arm_4", "right_arm_5", "right_arm_6"
    ]
    dof_ids_right = [model.joint(name).qposadr[0] for name in torso_right_joints]

    actuator_names_right = [
        "link1_act", "link2_act", "link3_act", "link4_act", "link5_act", "link6_act",
        "right_arm_1_act", "right_arm_2_act", "right_arm_3_act",
        "right_arm_4_act", "right_arm_5_act", "right_arm_6_act", "right_arm_7_act"
    ]
    actuator_ids_right = [model.actuator(a_name).id for a_name in actuator_names_right]
    right_finger_joint = "gripper_finger_r2"
    ee_site_name_right = "ee_site_r"

    home_manual = np.array([-0.013,-0.013,0.0017,-0.13,-0.0085,-0.0034,0.00053, 
    -6.8e-05,0.0011,8.6e-05,-1.4e-05,6.8e-05,-6.6e-06,9.5e-05,1.5e-07,6.7e-09,3.8e-07,
    0.00099,-0.0001,1.2e-05,6.9e-05,6.8e-06,8.6e-05,2.1e-07,-4e-07,3.5e-17,-1.2e-08,
    -3.5e-07,0.6,0.2,0.45,1.0,1.7e-19,0.0,0.0,0.6,-0.2,0.45,1.0,1.7e-19,0.0,0.0
    ], dtype=float)

    home_qpos_left  = home_manual[dof_ids_left].copy()
    home_qpos_right = home_manual[dof_ids_right].copy()

    cube_pos_left  = np.array([0.6,  0.2,  0.45])
    cube_pos_right = np.array([0.6, -0.2,  0.45])

    above_left   = cube_pos_left  + np.array([0.1,-0.05,0.10])
    descend_left = cube_pos_left  + np.array([0.1,-0.05,0])
    lift_left    = cube_pos_left  + np.array([0.0,0.0,0.30])

    above_right   = cube_pos_right + np.array([0.1, 0.05, 0.10])
    descend_right = cube_pos_right + np.array([0.1,0.05,0])
    lift_right    = cube_pos_right + np.array([0.0, 0.0, 0.30])

    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    current_target_left  = above_left.copy()
    current_target_right = above_right.copy()

    mujoco.mj_resetData(model, data)
    data.qpos[:] = home_manual
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.azimuth   = 180
        viewer.cam.elevation = -30
        viewer.cam.distance  = 2.0
        viewer.cam.lookat    = [0.3, 0.1, 0.0]

        print("Starting pick-and-lift with both arms in sync...")
        storage = initialize_hdf5_storage(data_dir="data", episode_idx=0)
        renderer = mujoco.Renderer(model, height=480, width=640)
        while viewer.is_running():
            t0 = time.time()

            if state != DONE:
                dq_left = compute_dq(
                    model, data, dof_ids_left, home_qpos_left,
                    ee_site_name_left, current_target_left, target_quat
                )
                dq_right = compute_dq(
                    model, data, dof_ids_right, home_qpos_right,
                    ee_site_name_right, current_target_right, target_quat
                )

                new_qpos = data.qpos.copy()

                dq_full_left = np.zeros(model.nv)
                dq_full_left[dof_ids_left] = dq_left
                mujoco.mj_integratePos(model, new_qpos, dq_full_left, INTEGRATION_DT)
                for act_id, joint_val in zip(actuator_ids_left, new_qpos[dof_ids_left]):
                    data.ctrl[act_id] = joint_val

                dq_full_right = np.zeros(model.nv)
                dq_full_right[dof_ids_right] = dq_right
                mujoco.mj_integratePos(model, new_qpos, dq_full_right, INTEGRATION_DT)
                for act_id, joint_val in zip(actuator_ids_right, new_qpos[dof_ids_right]):
                    data.ctrl[act_id] = joint_val

                mujoco.mj_step(model, data)

                site_id_l = model.site(ee_site_name_left).id
                site_id_r = model.site(ee_site_name_right).id
                dist_left  = np.linalg.norm(current_target_left  - data.site_xpos[site_id_l])
                dist_right = np.linalg.norm(current_target_right - data.site_xpos[site_id_r])
                print(dist_right,dist_left)
                if (dist_left < POS_THRESHOLD) and (dist_right < POS_THRESHOLD):
                    if state == ABOVE:
                        print("Both arms above cubes -> Opening grippers.")
                        set_gripper_position_left(model, data, 1.0)
                        set_gripper_position_right(model, data, 1.0)
                        left_ready  = is_finger_near_target(model, data, left_finger_joint, 1.0)
                        right_ready = is_finger_near_target(model, data, right_finger_joint, 1.0)
                        if left_ready and right_ready:
                            state = DESCEND
                            current_target_left  = descend_left
                            current_target_right = descend_right
                            print("Descending to cubes...")

                    elif state == DESCEND:
                        print("Both arms at cubes -> Closing grippers.")
                        set_gripper_position_left(model, data, 0.6)
                        set_gripper_position_right(model, data, 0.6)
                        left_ready  = is_finger_near_target(model, data, left_finger_joint, 0.6)
                        right_ready = is_finger_near_target(model, data, right_finger_joint, 0.6)
                        if left_ready and right_ready:
                            state = CLOSE_GRIP

                    elif state == CLOSE_GRIP:
                        print("Lifting cubes together...")
                        state = LIFT
                        current_target_left  = lift_left
                        current_target_right = lift_right

                    elif state == LIFT:
                        print("Dual-arm pick complete -> DONE.")
                        state = DONE
            store_data(storage, data, renderer)
            viewer.sync()
            elapsed = time.time() - t0
            if DT - elapsed > 0:
                time.sleep(DT - elapsed)


        print("Viewer closed. Exiting.")
    final_save(storage)


if __name__ == "__main__":
    main()
