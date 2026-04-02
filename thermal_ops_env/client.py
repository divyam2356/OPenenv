# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thermal Ops Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ThermalOpsAction, ThermalOpsObservation


class ThermalOpsEnv(
    EnvClient[ThermalOpsAction, ThermalOpsObservation, State]
):
    """Client for the Thermal Ops Env Environment."""

    def _step_payload(self, action: ThermalOpsAction) -> Dict:
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ThermalOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = ThermalOpsObservation(
            ambient_temp=obs_data.get("ambient_temp", 0.0),
            rack_temps=obs_data.get("rack_temps", []),
            power_loads=obs_data.get("power_loads", []),
            fan_rpms=obs_data.get("fan_rpms", []),
            chiller_setpoint=obs_data.get("chiller_setpoint", 0.0),
            broken_fans=obs_data.get("broken_fans", []),
            step_count=obs_data.get("step_count", 0),
            status_message=obs_data.get("status_message", ""),
            text_observation=obs_data.get("text_observation", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
