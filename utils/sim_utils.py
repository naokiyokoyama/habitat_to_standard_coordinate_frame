import numpy as np
import habitat_sim
import magnum as mn


def place_agent(sim):
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.1, 1.0]
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "None"
    backend_cfg.enable_physics = True

    camera_resolution = [int(540 * 1.35), int(720 * 1.35)]
    sensors = {
        "rgb_camera": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def load_robot(sim, urdf_path, fixed_base=False):
    ao_mgr = sim.get_articulated_object_manager()
    robot_id = ao_mgr.add_articulated_object_from_urdf(urdf_path, fixed_base=fixed_base)
    assert robot_id is not None, f"URDF failed to load from {urdf_path}! Aborting."
    return robot_id


def visualize_axes(sim, sphere_size=0.02, max_limit=2):
    """
    Each axis should have a different spacing of spheres.
    x-axis is the densest, followed by y-axis, and then z-axis.
    In habitat, the coord conventions are: [-y, z, -x]
    """
    obj_template_mgr = sim.get_object_template_manager()
    sphere_handle = obj_template_mgr.get_template_handles("sphereSolid")[0]
    sphere_template = obj_template_mgr.get_template_by_handle(sphere_handle)
    sphere_template.scale = np.array([sphere_size] * 3)
    obj_template_mgr.register_template(sphere_template)

    signs = [-1, 1, -1]
    sphere_density = [0.3, 0.1, 1.0]
    spheres = []
    rigid_obj_mgr = sim.get_rigid_object_manager()
    for i in range(3):
        num_spheres = int(sphere_density[i] * max_limit / sphere_size)
        for offset in np.linspace(0, max_limit, num_spheres):
            coord = mn.Vector3()
            coord[i] = offset * signs[i]

            # Spawn and translate sphere
            s = rigid_obj_mgr.add_object_by_template_handle(sphere_handle)
            s.translation = coord
            spheres.append(s)

    return spheres
