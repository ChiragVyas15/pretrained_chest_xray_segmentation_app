import os
import subprocess
from pathlib import Path

target_data_dir = Path(r"c:\Users\chira\OneDrive\Desktop\x RAY LUNG DISEASES\pretrained_chest_xray_segmentation_app\data\Lung Segmentation")
source_data_dir = Path(r"c:\Users\chira\OneDrive\Desktop\x RAY LUNG DISEASES\chest_xray_anatomical_segmentation\data\Lung Segmentation")

if not source_data_dir.exists():
    source_data_dir = Path(r"c:\Users\chira\OneDrive\Desktop\chexmask datset and model\Lung Segmentation")

print(f"Source data exists: {source_data_dir.exists()} ({source_data_dir})")

if not target_data_dir.exists() and source_data_dir.exists():
    cmd = f'cmd /c mklink /J "{target_data_dir}" "{source_data_dir}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Junction creation result:", res.stdout, res.stderr)

if target_data_dir.exists():
    print("Target dataset items:", [f.name for f in target_data_dir.iterdir()])
