import cv2
import habitat_sim
import magnum as mn
import numpy as np

from utils.geometry_utils import magnum_to_agent_state
from utils.sim_utils import make_configuration


def main(sim):
    obj_template_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # Visualize the coordinate frame using spheres.
    sphere_size = 0.02
    max_limit = 2  # meters

    sphere_handle = obj_template_mgr.get_template_handles("sphereSolid")[0]
    sphere_template = obj_template_mgr.get_template_by_handle(sphere_handle)
    sphere_template.scale = np.array([sphere_size] * 3)
    obj_template_mgr.register_template(sphere_template)

    # Each axis should have a different spacing of spheres.
    # x-axis is the densest, followed by y-axis, and then z-axis.
    # In habitat, the coord conventions are: [-y, z, -x]
    signs = [-1, 1, -1]
    sphere_density = [0.3, 0.1, 1.0]
    spheres = []
    for i in range(3):
        num_spheres = int(sphere_density[i] * max_limit / sphere_size)
        for offset in np.linspace(0, max_limit, num_spheres):
            coord = mn.Vector3()
            coord[i] = offset * signs[i]

            # Spawn and translate sphere
            s = rigid_obj_mgr.add_object_by_template_handle(sphere_handle)
            s.translation = coord
            spheres.append(s)

    agent = sim.initialize_agent(agent_id=0)
    global_T_camera_hab = mn.Matrix4().look_at(
        eye=mn.Vector3(-1.0, 1.0, -1.0) * 2,
        target=mn.Vector3(0, 0, 0),
        up=mn.Vector3(0, 1, 0),
    )
    agent.set_state(magnum_to_agent_state(global_T_camera_hab))

    img = sim.get_sensor_observations()["rgb_camera"]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Coordinate Frame", img_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    with habitat_sim.Simulator(make_configuration()) as sim:
        main(sim)
