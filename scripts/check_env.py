import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

pkgs = ['torch', 'torchvision', 'PIL', 'fastapi', 'uvicorn', 'librosa', 'facenet_pytorch', 'yaml']
print("\nPackage Status:")
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'installed')
        print(f"  - {p}: {ver}")
    except ImportError:
        print(f"  - {p}: MISSING")

print("\nCUDA / GPU Status:")
try:
    import torch
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    print(f"  - CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  - Error checking CUDA: {e}")

print("\nDatasets YAML & Path Check:")
yaml_path = "scripts/datasets.yaml"
if os.path.exists(yaml_path):
    print(f"  - Found {yaml_path}")
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
            datasets = cfg.get("datasets", {})
            for name, info in datasets.items():
                lpath = info.get("local_path")
                exists = os.path.exists(lpath) if lpath else False
                count = 0
                if exists and os.path.isdir(lpath):
                    for _, _, files in os.walk(lpath):
                        count += len(files)
                print(f"    * {name} ({info.get('name')}): local_path='{lpath}' | Exists: {exists} | Total Files: {count}")
    except Exception as e:
        print(f"  - Error parsing {yaml_path}: {e}")
else:
    print(f"  - MISSING: {yaml_path}")
