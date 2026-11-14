# LLM state preparation

`llm_state.prepare_llm_request(observation, config)` converts an existing
Gymnasium observation into versioned system/user messages and SHA256 hashes.
It uses only the observation, preserves all float32 values, validates shape,
range and finiteness, and adds lane/longitudinal coordinates. No rewards, engine
internals, future states or identities of nearby cars are added.

```python
from deeptraffic_llm_reward.llm_state import prepare_llm_request

observation, info = env.reset(seed=42)
request = prepare_llm_request(observation, env.config)
# request['messages'] is prepared for a later LLM client.
```

The benchmark produces 52 rows by 13 columns (676 values). Negative row offsets
are ahead; zero is the vehicle's longitudinal position. With patchesBehind=0,
offset zero is not included. Relative lane zero is the center column. Values
are the native engine values: free space is 1.0; occupied cells contain mph/2000.
Zero cannot distinguish stopped traffic
from off-road. Vehicle identities and separate ego speed are unavailable.

Actions are taken directly from the environment enum: maintain=0, accelerate=1,
decelerate=2, left=3, right=4. Earlier conversational examples using a different
numbering were incorrect; this module uses the actual engine numbering.

This module performs no HTTP requests, response parsing, action selection or
simulation steps. It generates no files unless the caller explicitly saves the
returned JSON. It requires no model download and changes no environment logic.
Any later shortening or semantic compression of the grid should be versioned
and evaluated separately; v1 retains the complete numerical observation.
