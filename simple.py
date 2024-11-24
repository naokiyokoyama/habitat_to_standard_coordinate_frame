import cv2
import habitat_sim
import magnum as mn

from utils.geometry_utils import magnum_to_agent_state
from utils.sim_utils import make_configuration, visualize_axes


def main(sim):
    visualize_axes(sim)

    camera = sim.initialize_agent(agent_id=0)
    global_T_camera_hab = mn.Matrix4().look_at(
        eye=mn.Vector3(-1.0, 1.0, -1.0) * 2,
        target=mn.Vector3(0, 0, 0),
        up=mn.Vector3(0, 1, 0),
    )
    camera.set_state(magnum_to_agent_state(global_T_camera_hab))

    img = sim.get_sensor_observations()["rgb_camera"]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Coordinate Frame", img_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    with habitat_sim.Simulator(make_configuration()) as sim:
        main(sim)
