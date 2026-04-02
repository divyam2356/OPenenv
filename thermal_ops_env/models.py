# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Thermal Ops Env Environment.
"""

from typing import Dict, Any, List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ThermalOpsAction(Action):
    """Action for the Thermal Ops environment - represents a tool call."""
    tool_name: str = Field(..., description="The name of the tool function to call (e.g. 'set_fan_speed', 'wait').")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")


class ThermalOpsObservation(Observation):
    """Observation from the Thermal Ops environment - represents thermodynamic state."""
    ambient_temp: float = Field(0.0, description="Ambient external temperature")
    rack_temps: List[float] = Field(default_factory=list, description="List of server rack temperatures")
    power_loads: List[float] = Field(default_factory=list, description="List of computational power loads per rack")
    fan_rpms: List[int] = Field(default_factory=list, description="List of fan speeds (RPM) per rack")
    chiller_setpoint: float = Field(0.0, description="Global chiller base temperature setpoint")
    broken_fans: List[int] = Field(default_factory=list, description="List of broken fan IDs")
    step_count: int = Field(0, description="Number of elapsed simulation steps")
    status_message: str = Field("", description="Status message of the last executed action")
    text_observation: str = Field("", description="String representation of the observation for LLM ingestion")
