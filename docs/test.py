import numpy as np
from deeptraffic_llm_reward import DeepTrafficConfig
from deeptraffic_llm_reward.llm_policy import LLMPolicyAdapter

class SimulatedClient:
    def chat(self, messages):
        return {
            'request': {'model': 'simulated', 'messages': messages},
            'response': {'done': True, 'message': {'content': '{"action":3}'}},
        }

config = DeepTrafficConfig(lanes_side=1, patches_ahead=2, patches_behind=1)
observation = np.ones(9, dtype=np.float32)
policy = LLMPolicyAdapter(config, SimulatedClient())
action = policy(observation)
print(action)                         # 3 = CHANGE_LEFT
print(policy.last_record['status'])   # ok
print(policy.last_record['resolution'])
print(policy.last_record['config'])