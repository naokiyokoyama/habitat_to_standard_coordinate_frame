import cv2
import habitat_sim
import magnum as mn
import numpy as np


from utils.geometry_utils import (
    set_robot_base_transform,
    magnum_to_agent_state,
    convert_conventions,
    get_robot_base_transform,
    constrain_quaternion,
    generate_random_quaternion,
)
from utils.sim_utils import load_robot, visualize_axes, make_configuration
from utils.visualization_utils import add_text_to_image

ROBOT_FILE = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"


def main(sim):
    spheres = visualize_axes(sim)

    # Spawn camera
    camera = sim.initialize_agent(agent_id=0)
    global_T_camera_std = mn.Matrix4().look_at(
        eye=mn.Vector3(1.0, 1.0, 1.0) * 2,
        target=mn.Vector3(0, 0, 0),
        up=mn.Vector3(0, 0, 1),
    )
    global_T_camera_hab = convert_conventions(global_T_camera_std, reverse=True)
    camera.set_state(magnum_to_agent_state(global_T_camera_hab))

    # Spawn robot
    robot = load_robot(sim, ROBOT_FILE)
    robot.joint_positions[:12] = [0.0, 0.7, -1.5] * 4  # Make legs look nice

    axes = np.eye(3)
    names = ["roll pitch yaw".split(), "xyz"]
    values = [
        np.linspace(np.radians(-60), np.radians(60), 150),
        np.linspace(-1.5, 1.5, 150),
    ]
    window_names = ["Roll Pitch Yaw test", "XYZ test"]
    # for idx in range(2):
    for idx in range(0):
        for axis_idx in range(3):
            for val in values[idx]:
                global_T_base_std = (
                    mn.Matrix4.rotation(mn.Rad(val), mn.Vector3(axes[axis_idx]))
                    if idx == 0
                    else mn.Matrix4.translation(axes[axis_idx] * val)
                )
                set_robot_base_transform(robot, global_T_base_std)

                img = sim.get_sensor_observations()["rgb_camera"]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                unit = "degrees" if idx == 0 else "meters"
                if idx == 0:
                    val = np.degrees(val)
                img_bgr = add_text_to_image(
                    img_bgr, f"{names[idx][axis_idx]}: {val:.2f} {unit}"
                )
                cv2.imshow(window_names[idx], img_bgr)
                cv2.waitKey(5)
        cv2.destroyAllWindows()

    goal_w, goal_v = generate_random_quaternion()
    offsets = [convert_conventions(s.translation) for s in spheres]
    while True:
        q_a = mn.Quaternion().from_matrix(get_robot_base_transform(robot).rotation())
        q_a = (q_a.scalar, np.array(q_a.vector))
        w, v = constrain_quaternion(q_a, (goal_w, goal_v), np.radians(2))
        new_base_tf = mn.Matrix4().from_(
            mn.Quaternion(mn.Vector3(v), w).to_matrix(),
            mn.Vector3(),
        )
        set_robot_base_transform(robot, new_base_tf)
        global_T_base_std = get_robot_base_transform(robot)

        for sphere, offset in zip(spheres, offsets):
            global_sphere_pos_std = global_T_base_std.transform_point(offset)
            sphere.translation = convert_conventions(
                global_sphere_pos_std, reverse=True
            )

        img = sim.get_sensor_observations()["rgb_camera"]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bgr = add_text_to_image(img_bgr, "Press 'q' to quit.")
        cv2.imshow("ee test", img_bgr)
        k = cv2.waitKey(10)
        if k == ord("q"):
            quit()
        if w == goal_w and np.array_equal(v, goal_v):
            goal_w, goal_v = generate_random_quaternion()


if __name__ == "__main__":
    with habitat_sim.Simulator(make_configuration()) as sim:
        main(sim)
