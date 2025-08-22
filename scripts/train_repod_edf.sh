# scripts/train_repod_edf.sh
#!/usr/bin/env bash
set -e
python -u -m src.repod_edf.train --config configs/repod_edf_config.yaml