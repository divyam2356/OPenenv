# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Thermal Ops Env Environment Implementation.
"""

import json
import random
from typing import Set, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ThermalOpsAction, ThermalOpsObservation
except ImportError:
    from models import ThermalOpsAction, ThermalOpsObservation


class ThermalOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the thermal_ops_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self.num_racks = 3
        self.max_steps = 10
        self.w1_energy = 0.5
        self.w2_overheat = 1000.0
        self.safe_temp_max = 25.0
        self.critical_temp = 27.0
        
        # State variables
        self.ambient_temp: float = 24.0
        self.rack_temps: List[float] = []
        self.power_loads: List[float] = []
        self.fan_rpms: List[int] = []
        self.chiller_setpoint: float = 20.0
        self.energy_cost: float = 0.15
        self.total_energy_consumed: float = 0.0
        self.broken_fans: Set[int] = set()
        self._done: bool = False
        self._reward: float = 0.0

        self.reset()
        
    def reset(self) -> ThermalOpsObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self.ambient_temp = random.uniform(20.0, 28.0)
        self.rack_temps = [random.uniform(20.0, 26.0) for _ in range(self.num_racks)]
        self.power_loads = [random.uniform(5.0, 20.0) for _ in range(self.num_racks)]
        self.fan_rpms = [random.choice([500, 1000, 1500, 2000]) for _ in range(self.num_racks)]
        self.chiller_setpoint = random.uniform(10.0, 20.0)
        self.energy_cost = random.uniform(0.10, 0.25)
        self.total_energy_consumed = 0.0
        self.broken_fans = set(i for i in range(self.num_racks) if random.random() < 0.3)
        self._reward = 0.0
        self._done = False
        
        return self._get_obs(status="Thermal Ops Environment ready!")

    def _get_obs(self, status: str = "", step_reward: float = 0.0) -> ThermalOpsObservation:
        obs_dict = {
            "ambient_temp": self.ambient_temp,
            "rack_temps": [round(t, 2) for t in self.rack_temps],
            "power_loads": [round(l, 2) for l in self.power_loads],
            "fan_rpms": self.fan_rpms,
            "chiller_setpoint": round(self.chiller_setpoint, 2),
            "broken_fans": list(self.broken_fans),
            "step_count": self._state.step_count
        }
        
        return ThermalOpsObservation(
            ambient_temp=self.ambient_temp,
            rack_temps=[round(t, 2) for t in self.rack_temps],
            power_loads=[round(l, 2) for l in self.power_loads],
            fan_rpms=list(self.fan_rpms),
            chiller_setpoint=round(self.chiller_setpoint, 2),
            broken_fans=list(self.broken_fans),
            step_count=self._state.step_count,
            status_message=status,
            text_observation=f"Observation: {json.dumps(obs_dict)}\\nStatus: {status}",
            done=self._done,
            reward=step_reward,
            metadata={"total_energy": self.total_energy_consumed}
        )

    def step(self, action: ThermalOpsAction) -> ThermalOpsObservation:
        """Execute a step based on the tool call."""
        if self._done:
            return self._get_obs(status="Episode over.", step_reward=0.0)

        tool = action.tool_name
        args = action.arguments
        status = "Unknown tool or invalid arguments."
        step_reward = 0.0

        if tool == "set_fan_speed":
            rack_id = args.get("rack_id")
            rpm = args.get("rpm")
            if isinstance(rack_id, int) and isinstance(rpm, int):
                if 0 <= rack_id < self.num_racks and rack_id not in self.broken_fans:
                    self.fan_rpms[rack_id] = max(0, min(5000, rpm))
                    status = f"Fan {rack_id} speed bounded and set to {self.fan_rpms[rack_id]} RPM."
                else:
                    status = "Failed: Invalid rack_id or fan is broken."
                    
        elif tool == "adjust_chiller":
            chiller_temp = args.get("chiller_temp")
            if chiller_temp is not None:
                self.chiller_setpoint = max(5.0, min(30.0, float(chiller_temp)))
                status = f"Chiller setpoint adjusted to {self.chiller_setpoint}°C."
                
        elif tool == "migrate_workload":
            source_rack = args.get("source_rack")
            target_rack = args.get("target_rack")
            if isinstance(source_rack, int) and isinstance(target_rack, int):
                if 0 <= source_rack < self.num_racks and 0 <= target_rack < self.num_racks and source_rack != target_rack:
                    load_to_move = self.power_loads[source_rack] * 0.5
                    self.power_loads[source_rack] -= load_to_move
                    self.power_loads[target_rack] += load_to_move
                    status = f"Migrated {load_to_move:.2f} workload from rack {source_rack} to rack {target_rack}."
                else:
                    status = "Failed: Invalid source or target rack."
                    
        elif tool == "wait":
            energy_consumed = 0.0
            overheat_penalty_total = 0.0
            
            chiller_delta = max(0, self.ambient_temp - self.chiller_setpoint)
            energy_consumed += 0.5 * (chiller_delta ** 2)
            
            for i in range(self.num_racks):
                heat_generated = 0.1 * self.power_loads[i]
                rpm = self.fan_rpms[i] if i not in self.broken_fans else 0
                cooling_power = (rpm / 1000.0) * 0.5
                chiller_effect = max(0, self.rack_temps[i] - self.chiller_setpoint) * 0.1
                ambient_effect = (self.ambient_temp - self.rack_temps[i]) * 0.05
                
                self.rack_temps[i] += heat_generated - cooling_power - chiller_effect + ambient_effect
                energy_consumed += ((rpm / 1000.0) ** 3) * 0.2

                baseline_drift_penalty = abs(self.rack_temps[i] - 22.0) * 0.05
                overheat_penalty_total += baseline_drift_penalty
                
                if self.rack_temps[i] > self.safe_temp_max:
                    if self.rack_temps[i] > self.critical_temp:
                        overheat_penalty_total += self.w2_overheat
                    else:
                        overheat_penalty_total += self.w2_overheat * 0.1 * (self.rack_temps[i] - self.safe_temp_max)
                        
            cost = energy_consumed * self.energy_cost
            self.total_energy_consumed += energy_consumed
            
            # Sub-step reward
            step_reward = -(self.w1_energy * cost) - overheat_penalty_total
            self._reward += step_reward
            
            self._state.step_count += 1
            if self._state.step_count >= self.max_steps:
                self._done = True
                
            status = "Simulation step progressed."

        return self._get_obs(status=status, step_reward=step_reward)

    @property
    def state(self) -> State:
        return self._state
