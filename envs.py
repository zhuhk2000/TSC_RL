import os
import sys
import time

from typing import List, Dict, Literal
from collections import defaultdict

import gymnasium as gym
import traci
import sumolib
import numpy as np
import pandas as pd
import torch


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

SUMO_BIN = sumolib.checkBinary('sumo')
SUMO_GUI = sumolib.checkBinary('sumo-gui')




class TrafficLight:
    def __init__(self, tls: sumolib.net.TLS):
        """初始化交叉路口对象
        Args:
            tls: sumolib.net.TLS对象
        """
        self.tls = tls
        self.tls_id = self.tls.getID()
        # 1.get the phase of the traffic light, the length of the self.phases can be get to initialize the model output
        self.phases = [p.state for p in self.tls.getPhases()][::2]
        # 2.get the number of phases
        self.num_phases = len(self.phases)
        # 3.get in lanes
        self.inlines: List[sumolib.net.lane.Lane] = list(set(c[0] for c in self.tls.getConnections()))
        self.sorted_inlane = sorted(self.inlines, key=lambda lane: lane.getShape()[1])
        self.sorted_inlane_id = [lane.getID() for lane in self.sorted_inlane]
        self.sorted_inlane_len = np.array([lane.getLength() for lane in self.sorted_inlane])
        # 4.get out lanes
        self.outlanes: List[sumolib.net.lane.Lane] = list(set(c[1] for c in self.tls.getConnections()))
        self.sorted_outlane = sorted(self.outlanes, key=lambda lane: lane.getShape()[0])
        self.sorted_outlane_id = [lane.getID() for lane in self.sorted_outlane]
        self.sorted_outlane_len = np.array([lane.getLength() for lane in self.sorted_outlane])
        # 5.get connections
        connection_info: Dict[str, List] = defaultdict(list)
        for inlane, inlane_id in zip(self.sorted_inlane, self.sorted_inlane_id):
            for connection in inlane.getOutgoing():
                outlane_id = connection.getToLane().getID()
                connection_index = connection.getLinkIndex()
                direction = connection.getDirection()
                connection_info["inlane_id"].append(inlane_id)
                connection_info["inlane_index"].append(self.sorted_inlane.index(inlane_id))
                connection_info["connection_index"].append(connection_index)
                connection_info["direction"].append(direction)
                connection_info["outlane_id"].append(outlane_id)
                connection_info["outlane_index"].append(self.sorted_outlane.index(outlane_id))
        self.connection_info = pd.DataFrame(connection_info).set_index("connection_index").sort_index()
        # 6.prepare for homogenous check
        self.inlane_phase = defaultdict(set)
        for i, inlane_id in enumerate(self.sorted_inlane_id):
            connection_index = self.connection_info[self.connection_info["inlane_id"] == inlane_id].index
            for phase_index, phase in enumerate(self.phases):
                is_all_allowed = True
                for connection_index in connection_index:
                    if phase[connection_index] not in "Gg":
                        is_all_allowed = False
                        break
                if is_all_allowed:
                    self.inlane_phase[i].add(phase_index)
    
    def get_all_lane_vehicle_count(self):
        """获取所有车道的车辆数"""
        inlane_vehicle_count = np.array([traci.lane.getLastStepVehicleNumber(inlane_id) for inlane_id in self.sorted_inlane_id], dtype=np.float32)
        outlane_vehicle_count = np.array([traci.lane.getLastStepVehicleNumber(outlane_id) for outlane_id in self.sorted_outlane_id], dtype=np.float32)
        return inlane_vehicle_count, outlane_vehicle_count

    def get_all_lane_queue_count(self):
        inlane_queue_count = np.array([traci.lane.getLastStepHaltingNumber(inlane_id) for inlane_id in self.sorted_inlane_id], dtype=np.float32)
        outlane_queue_count = np.array([traci.lane.getLastStepHaltingNumber(outlane_id) for outlane_id in self.sorted_outlane_id], dtype=np.float32)
        return inlane_queue_count, outlane_queue_count
    
    def get_all_inline_effective_range_count(self, effective_range: float):
        """获取进口车道的有效范围内的车辆数"""
        in_lane_count = np.zeros(len(self.sorted_inlane_id), dtype=np.float32)
        for i, inlane_id in enumerate(self.sorted_inlane_id):
            for vehicle_id in traci.lane.getLastStepVehicleIDs(inlane_id):
                # vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
                # if vehicle_pos <= effective_range:
                #     in_lane_count[i] += 1
                path = traci.vehicle.getNextTLS(vehicle_id)
                if path is not None and path[0] == self.tls_id:
                    if path[0][2] <= effective_range:
                        in_lane_count[i] += 1
        return in_lane_count
    
    def get_phase_feature(self, feature_type: Literal["pressure", "demand", "efficient pressure"]):
        if feature_type == "pressure":
            inlane_vehicle_count, outlane_vehicle_count = self.get_all_lane_vehicle_count()
        elif feature_type == "demand":
            inlane_queue_count, outlane_queue_count = self.get_all_lane_queue_count()
        elif feature_type == "efficient pressure":
            inlane_vehicle_count = self.get_all_lane_vehicle_count()
            outlane_vehicle_count = np.zeros(len(self.sorted_outlane_id), dtype=np.float32)
        
        # 计算每个连接的压力
        inlane_index = self.connection_info["inlane_index"].tolist()
        outlane_index = self.connection_info["outlane_index"].tolist()
        connection_pressure = inlane_vehicle_count[inlane_index] - outlane_vehicle_count[outlane_index]

        # 计算每个相位的压力
        phase_pressure = np.zeros(len(self.phases), dtype=np.float32)
        for phase_index, phase in enumerate(self.phases):
            for connection_index in self.connection_info.index:
                if phase[connection_index] in "Gg":
                    phase_pressure[phase_index] += connection_pressure[connection_index]

        return phase_pressure
    
    def get_all_lane_feature(self, num_seg: int):
        num_inlane = len(self.sorted_inlane_id)
        inlane_feature = np.zeros((num_inlane, 2*num_seg), dtype=np.float32)
        inlane_seg_len = self.sorted_inlane_len / num_seg
        queue_count = 0.
        speed_loss_count = 0.

        for i, lane_id, seg_length in zip(range(num_inlane), self.sorted_inlane_id, inlane_seg_len):
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                distance = traci.vehicle.getLanePosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                seg_index = min(int(distance // seg_length), num_seg - 1)
                inlane_feature[i, 2*seg_index] += 1
                inlane_feature[i, 2*seg_index + 1] += speed
                if speed <= 1:
                    queue_count += 1
                speed_loss_count += max(min((20 - speed) / 20, 1.0), 0)
        inlane_feature[i, 1::2] /= inlane_feature[i, ::2] + 1e-5

        num_outlane = len(self.sorted_outlane_id)
        outlane_feature = np.zeros((num_outlane, 2*num_seg), dtype=np.float32)
        for i, lane_id, seg_length in zip(range(num_outlane), self.sorted_outlane_id, inlane_seg_len):
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                distance = traci.vehicle.getLanePosition(vehicle_id)
                seg_index = min(int(distance // seg_length), num_seg - 1)
                outlane_feature[i, 2*seg_index] += 1
                outlane_feature[i, 2*seg_index + 1] += traci.vehicle.getSpeed(vehicle_id)
        outlane_feature[i, 1::2] /= outlane_feature[i, ::2] + 1e-5

        return np.concatenate([inlane_feature, outlane_feature], axis=0).reshape(-1), queue_count/num_inlane, speed_loss_count/num_inlane


class SumoEnv(gym.Env):
    def __init__(self,
                 file_dir: str,
                 gui: bool = False,
                 delta_simu_steps: int = 5,
                 total_simu_steps: int = 3600,
                 with_phase: bool = False):
        self.file_dir = file_dir
        self.gui = gui
        self.delta_simu_steps = delta_simu_steps
        self.total_simu_steps = total_simu_steps
        self.with_phase = with_phase

        # 读取配置文件
        files = os.listdir(file_dir)
        self.sumocfg = os.path.join(file_dir, [f for f in files if f.endswith(".sumocfg")][0])
        self.net_file = os.path.join(file_dir, [f for f in files if f.endswith(".net.xml")][0])
        self.route_file = os.path.join(file_dir, [f for f in files if f.endswith(".rou.xml")][0])


        # 读取路网文件
        self.net = sumolib.net.readNet(self.net_file)
        self.tls_list: List[sumolib.net.TLS] = self.net.getTrafficLights()  # 通过路网文件获取的交叉路口对象列表
        self.tls_id_list = [tls.getID() for tls in self.tls_list]
        self.traffic_lights: List[TrafficLight] = [TrafficLight(tls) for tls in self.tls_list]  # 自己定义的TrafficLight对象列表
        # first_inter = self.idx2intersection[0]
        # first_in_lane2phase = first_inter.in_lane2phase
        # for inter in self.idx2intersection[1:]:
        #     assert first_in_lane2phase == inter.in_lane2phase
        self.current_phases: Dict[str, int] = {tls_id: 0 for tls_id in self.tls_id_list}
        self.simu_step = 0
        self.arrived_vehicles = 0
        self.vehicles: Dict[str, List[float, float]] = dict()

        self.action_spaces = [gym.spaces.Discrete(tls.num_phases) for tls in self.traffic_lights]
    
    def create_simulation(self, random_seed: int = 0):
        t = str(time.time())
        if self.is_render:
            sumo_cmd = [SUMO_GUI, '-c', self.sumocfg, "-W", "True", "--seed", random_seed]
        else:
            sumo_cmd = [SUMO_BIN, '-c', self.config_filepath, "-W", "True"]
        traci.start(sumo_cmd, label=f"{os.getpid()}_{self.label}_{t}")
        
    def reset(self, initial_random_steps: int, random_seed: int = 0):
        self.simu_step = 0
        self.arrived_vehicles = 0
        self.vehicles = dict()
        self.create_simulation(random_seed)
        self.current_phases = {tls_id: 0 for tls_id in self.tls_id_list}

        for tls_id, tls in zip(self.tls_id_list, self.traffic_lights):
            traci.trafficlight.setRedYellowGreenState(tls_id, tls.phases[0])
        
        for _ in range(initial_random_steps):
            self.step([action_space.sample() for action_space in self.action_spaces])
        
        obs = self.get_observation()
        reward = 0
        info = dict()
        info["reward"] = reward
        info["action"] = np.array(self.current_phases, dtype=np.int32)

        return obs, info
    
    def step(self, actions):
        # step yellow phase
        for tls, new_phase, current_phase in zip(self.traffic_lights, actions, self.current_phases.values()):
            if new_phase != current_phase:
                traci.trafficlight.setRedYellowGreenState(tls.tls_id, tls.phases[new_phase])
                self.current_phases[tls.tls_id] = new_phase
            else:
                traci.trafficlight.setRedYellowGreenState(tls.tls_id, tls.phases[current_phase])

        self.simu_step += 1
        traci.simulationStep()
        obs = self.get_observation()
        reward = self.get_reward()
        info = dict()
        info["reward"] = reward
        info["action"] = np.array(self.current_phases, dtype=np.int32)


    def get_observation(self):
        obs = []
        for tls in self.traffic_lights:
            in_lane_queue_count, _ = tls.get_all_lane_queue_count()
            obs.append(in_lane_queue_count)
        return obs
    
    def get_reward(self):
        pass
    
        
        
        
        