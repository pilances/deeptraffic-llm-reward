"""Load only the audited engine from an external, pinned checkout."""
import hashlib
import importlib.util
import os
from pathlib import Path

REVISION = 'acd7c2c9157f5b480e15cf0bcd2c793de45c5b68'
ENGINE_SHA256 = '17727a936beb0f76af29e09746bd5f0631a2057bb8b8e4fd434aa700e247c254'
JS_SHA256 = '661c1e6e23c337506d6588d76fe4fd9f679b9d266f0dc6a68b8fd4e9b674bf86'

def backend_root():
    return Path(os.environ.get('DEEPTRAFFIC_BACKEND',
        Path(__file__).resolve().parents[2] / '.external' / 'quillstacks'))

def engine_class():
    root = backend_root()
    path = root / 'src/deeptraffic/engine.py'
    for file, digest in ((path, ENGINE_SHA256),
            (root / 'reference/original_js/gameopt.js', JS_SHA256)):
        if not file.exists():
            raise RuntimeError('Run: python scripts/setup_backend.py')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Backend integrity check failed: {file}')
    spec = importlib.util.spec_from_file_location('_audited_deeptraffic_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine
