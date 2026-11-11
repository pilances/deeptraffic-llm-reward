"""Fetch a pinned external dependency without copying it into our distribution."""
import subprocess
from deeptraffic_llm_reward.backend import backend_root, engine_class, REVISION

root = backend_root()
if not root.exists():
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/Quillstacks/DeepTrafficByLLM', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--detach', REVISION], check=True)
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if head != REVISION:
    raise RuntimeError('Existing backend has another revision; leave it intact and choose DEEPTRAFFIC_BACKEND.')
engine_class()
print(f'Backend verified: {REVISION} at {root}')
