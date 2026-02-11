# Repository Guidelines

## Project Structure & Module Organization
- `mmdet3d/`: core Python package (models, datasets, runners, CUDA ops).
- `configs/`: experiment configs grouped by dataset/task (`simbev/`, `simbev2nuscenes/`, `nuscenes/`).
- `tools/`: entry-point scripts for training, evaluation, visualization, export, benchmarking, and data conversion.
- `tools/data_converter/`: dataset conversion utilities.
- `docker/`: containerized development environment.
- `assets/`: README media.
- `data/` and `pretrained/` (local, typically untracked): datasets and checkpoints.

## Build, Test, and Development Commands
- `python setup.py develop`: install in editable mode and build required extensions.
- `torchpack dist-run -np 8 python tools/train.py <config>`: distributed training.
- `torchpack dist-run -np 8 python tools/test.py <config> <ckpt> --eval bbox|map`: evaluation.
- `torchpack dist-run -np 8 python tools/visualize.py <config> --mode pred-simbev --checkpoint <ckpt> --split test --out-dir viz/<run>`: qualitative outputs.
- `python tools/create_data.py ...`: generate dataset metadata via converter scripts.
- `flake8 mmdet3d tools` and `isort mmdet3d tools`: lint/import-order checks before PR.

## Coding Style & Naming Conventions
- Python, 4-space indentation, max line length `120` (`setup.cfg`).
- Follow existing MMDetection3D-style module layout and registry patterns.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes.
- Keep config names descriptive and compositional (example: `configs/simbev/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`).

## Testing Guidelines
- There is no broad unit-test suite in this fork; validate changes with task-level runs.
- Minimum check for model changes: run one short `tools/test.py` evaluation on affected config.
- If touching CUDA/custom ops, add or update focused tests near the modified op (for example under `mmdet3d/ops/...`) and document how to execute them.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit subjects (for example: `Added multi-sweep support.`, `Fixed error in TransFusion head.`).
- Prefer format: `<scope>: <imperative summary>` (example: `seg: tune lidar centerpoint config`).
- PRs should include:
  - what changed and why,
  - configs/checkpoints used for validation,
  - key metrics or qualitative outputs (screenshots/paths in `viz/` when relevant),
  - linked issue(s) if applicable.
