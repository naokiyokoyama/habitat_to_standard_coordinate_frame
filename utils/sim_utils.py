import cv2
import math
import habitat_sim
import magnum as mn
import numpy as np

from typing import Union

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