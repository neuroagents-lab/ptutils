# ptutils
Pytorch utilities for model training. 

# Installation
To install run:
```
git clone https://github.com/anayebi/ptutils
cd ptutils/
pip install -e .
```

# Training
The example scripts support training ResNet-18 on ImageNet categorization, e.g.
```
cd ptutils/model_training/
python runner.py --config=configs/resnet18_supervised_imagenet_trainer_tpu.json
```
You can substitute your own training method by importing `from ptutils.model_training.runner import Runner`, and subclassing `Runner.train()`.

# Code Formatting:
Put this in `.git/hooks/pre-commit`, and run `sudo chmod +x .git/hooks/pre-commit`.

```
#!/usr/bin/env bash
  
echo "# Running pre-commit hook"
echo "#########################"

echo "Checking formatting"

format_occurred=false
declare -a black_dirs=("ptutils/" "setup.py")
for black_dir in "${black_dirs[@]}"; do
    echo ">>> Checking $black_dir"
    black --check "$black_dir"

    if [ $? -ne 0 ]; then
        echo ">>> Reformatting now!"
        black "$black_dir"
        format_occurred=true
    fi
done

if [ "$format_occurred" = true ]; then
    exit 1
fi
```
