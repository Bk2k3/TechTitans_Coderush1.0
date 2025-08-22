# scripts/train_kaggle_img.sh
#!/usr/bin/env bash
set -e
python -u -m src.kaggle_img.train --config configs/kaggle_img_config.yaml
