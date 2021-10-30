import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.
    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 20,  # [s]
            "policy_frequency": 5,  # [Hz]

            "COLLISION_REWARD": 200,
            "HIGH_SPEED_REWARD": 1,
            "HEADWAY_COST": 4,
            "HEADWAY_TIME": 1.2,
            "MERGING_LANE_COST": 4,

            "ego_spacing": 2,
            "vehicles_density": 1,
            # "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            # "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "traffic_density": 1

        })
        # config.update({
        #     "observation": {
        #         "type": "Kinematics"},
        #     "action": {
        #         "type": "DiscreteMetaAction",
        #         "longitudinal": True,
        #         "lateral": True},
        #     "controlled_vehicles": 1,
        #     "screen_width": 600,
        #     "screen_height": 120,
        #     "centering_position": [0.3, 0.5],
        #     "scaling": 3,
        #     "simulation_frequency": 15,  # [Hz]
        #     "duration": 20,  # time step
        #     "policy_frequency": 5,  # [Hz]
        #     "reward_speed_range": [20, 30],
        #     "collision_reward": 200,
        #     "high_speed_reward": 1,
        #     "HEADWAY_COST": 4,
        #     "HEADWAY_TIME": 1.2,
        #     "MERGING_LANE_COST": 4,
        #     "traffic_density": 1
        # })
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4
        })
        return config

    def _reward(self, action: int) -> float:
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    # modified
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # if vehicle.lane_index == ("b", "c", 1):
        #     Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
        #             10 * self.ends[2]))
        # else:
        Merging_lane_cost = 0

        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
            #         "c", "d", 0):
            #     v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
            #     if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
            #         v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
            #                                                     self.road.network.side_lanes(
            #                                                         vehicle.lane_index)[
            #                                                         0])
            #     elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:
            #         v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
            #     else:
            #         v_fr, v_rr = None, None
            # else:
            #     v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
            #     if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
            #         v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
            #                                                     self.road.network.side_lanes(
            #                                                         vehicle.lane_index)[0])
            #     elif vehicle.lane_index == ("k", "b", 0):
            #         v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
            #     else:
            #         v_fl, v_rl = None, None
            v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                        self.road.network.side_lanes(
                                                        vehicle.lane_index)[0])
            
            for v in [v_fr, vehicle, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=0) -> None:

        self._create_road()

        if self.config["traffic_density"] == 1:
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(1, 4), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(1, 4), 1)[0]

        elif self.config["traffic_density"] == 2:
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        self._create_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

    #     self.controlled_vehicles = []
    #     for others in other_per_controlled:
    #         controlled_vehicle = self.action_type.vehicle_class.create_random(
    #             self.road,
    #             speed=25,
    #             lane_id=self.config["initial_lane_id"],
    #             spacing=self.config["ego_spacing"]
    #         )
    #         self.controlled_vehicles.append(controlled_vehicle)
    #         self.road.vehicles.append(controlled_vehicle)

    #         for _ in range(others):
    #             vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

    def _create_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        # print("CAV and HDV count: ", num_CAV, num_HDV)
        spawn_points_s = [10, 50, 90, 130, 170, 210, 250, 290, 330, 370, 410, 450]
        # spawn_points_s = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        # spawn_points_s = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310]
        # spawn_points_s = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210]
        # spawn_points_m = [5, 45, 85, 125, 165, 205, 245, 285]

        """Spawn points for CAV"""
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV, replace=False)
        # spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
        #                                    replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        # spawn_point_m_c = list(spawn_point_m_c)
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        # for b in spawn_point_m_c:
        #     spawn_points_m.remove(b)
        # print(len(spawn_points_s))
        # print(num_HDV)

        """Spawn points for HDV"""
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV, replace=False)
        # spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2,
        #                                    replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        # spawn_point_m_h = list(spawn_point_m_h)

        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 27
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("0", "1", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        # """spawn the rest CAV on the merging road"""
        # for _ in range(num_CAV - num_CAV // 2):
        #     ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
        #         spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
        #     self.controlled_vehicles.append(ego_vehicle)
        #     road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("0", "1", 0)).position(
                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

        # """spawn the rest HDV on the merging road"""
        # for _ in range(num_HDV - num_HDV // 2):
        #     road.vehicles.append(
        #         other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
        #             spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
        #                             speed=initial_speed.pop(0)))

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)