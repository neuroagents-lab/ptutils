# ptutils
Pytorch utilities for model training on GPU and TPU in a single, flexible interface that can be subclassed with your own methods.
Model results are (optionally) saved to a MongoDB, for asynchronous visualization.

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
python runner.py --config=configs/resnet18_supervised_imagenet_trainer_[gpu/tpu].json
```
You can substitute your own training method by importing `from ptutils.model_training.runner import Runner`, and subclassing `Runner.train()`.

# MongoDB
By default, this packages saves model results to MongoDB.
If you would like to use it, follow [these instructions](https://www.mongodb.com/docs/manual/installation/) to install MongoDB.
Otherwise, to disable this feature, set `"use_mongodb": false` in your configuration json.

The function `ptutils.core.utils.grab_results()` is an example of how to grab the results from MongoDB for the `SupervisedImageNetTrainer`, and [this notebook](https://github.com/anayebi/ptutils/blob/main/Plot%20Model%20Training%20Results%20Example.ipynb) gives an example of plotting it.

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

# Contributors
- [Aran Nayebi](https://github.com/anayebi) (Stanford/MIT)
- [Nathan C. L. Kong](https://github.com/nathankong) (Stanford)
- [Javier Sagastuy-Brena](https://github.com/jvrsgsty) (Stanford)

# License
MIT
