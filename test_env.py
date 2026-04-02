import time
from thermal_ops_env.server.thermal_ops_env_environment import ThermalOpsEnvironment
from thermal_ops_env.models import ThermalOpsAction

def test_env():
    print("Initializing environment...")
    env = ThermalOpsEnvironment()
    
    obs = env.reset()
    print(f"Initial State:\n{obs.text_observation}")
    
    # Send a tool call
    action = ThermalOpsAction(
        tool_name="set_fan_speed",
        arguments={"rack_id": 0, "rpm": 2000}
    )
    print("\nSending set_fan_speed action...")
    obs = env.step(action)
    print(f"Status: {obs.status_message}")
    
    # Send wait tool call
    action = ThermalOpsAction(
        tool_name="wait",
        arguments={}
    )
    print("\nSending wait action...")
    obs = env.step(action)
    print(f"Status: {obs.status_message}")
    print(f"Post-wait State:\n{obs.text_observation}")
    print(f"Reward: {obs.reward}")

if __name__ == '__main__':
    test_env()
