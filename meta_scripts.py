import os

# Automatically generate tree structure
def generate_tree(path, exclude=None, level=0):
    if exclude is None:
        exclude = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if any(excluded in item_path for excluded in exclude):
            continue
        print(" " * (level * 4) + "|-- " + item)
        if os.path.isdir(item_path):
            generate_tree(item_path, exclude, level + 1)

# Generate tree
generate_tree(".", exclude=[".venv", "__pycache__", ".vscode", ".git"])