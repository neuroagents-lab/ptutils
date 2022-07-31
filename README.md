# ptutils
Pytorch utilities for model training

# Formatting:
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
