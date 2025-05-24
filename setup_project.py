# setup_project.py
import os

folders = ['data', 'env', 'model', 'train']
for folder in folders:
    init_path = os.path.join(folder, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('# Package initializer\n')
        print(f"✔️ Created: {init_path}")
    else:
        print(f"ℹ️ Already exists: {init_path}")
