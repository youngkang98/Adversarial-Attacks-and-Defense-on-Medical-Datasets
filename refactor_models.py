import os
import re
from pathlib import Path

models_dir = Path('src/models')

import_block = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

"""

# Regex patterns to find hardcoded paths
patterns = {
    # Models (.pth, .pth.tar)
    r"['\"](?:\.\./[^'\"]+|C:/Users/[^'\"]+|C:\\Users\\[^'\"]+)/.*?(\w+\.pth(?:\.tar)?)['\"]": lambda m: "str(config.get_model_path('{}'))".format(m.group(1)),
    
    # Specific known model files or checkpoints
    r"['\"]C:/Users/[^'\"]+/checkpoint\.pth['\"]": "str(config.get_model_path('checkpoint.pth'))",
    r"['\"]C:/Users/[^'\"]+/classifier_checkpoint\.pth['\"]": "str(config.get_model_path('classifier_checkpoint.pth'))",
    r"['\"]epoch10_model\.pth['\"]": "str(config.get_model_path('epoch10_model.pth'))",
    r"['\"]OCT\.pth\.tar['\"]": "str(config.get_model_path('OCT.pth.tar'))",
    
    # Datasets - ISIC / CSV
    r"['\"]C:/Users/[^'\"]+/New UAP/(ISIC2019_.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/New UAP/(CXRAY-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/New UAP/(OCT2017-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/(df_train\.csv|df_val\.csv|df_test\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    
    # Image Directories
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_Training_Input/?(?:ISIC_2019_Training_Input/)?['\"]": "str(config.get_data_path('ISIC2019'))",
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_train/?['\"]": "str(config.get_data_path('ISIC2019'))",
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_test/?['\"]": "str(config.get_data_path('ISIC2019'))",
    
    # OCT Directories
    r"['\"]C:/Users/[^'\"]+/archive/OCT2017/?['\"]": "str(config.get_data_path('OCT'))",
    r"['\"]C:/Users/[^'\"]+/archive/OCT2017/(train|test|val)/?['\"]": lambda m: "str(config.get_data_path('OCT/{}'))".format(m.group(1)),
    
    # Results / Output files - using regex for savefig
    r"plt\.savefig\s*\(\s*f?['\"]([^'\"]+?\.png)['\"]": r"plt.savefig(str(config.get_experiment_path(f'\1'))",
    
    # Classification reports and eval results
    r"['\"](classfication_report.*?\.txt)['\"]": lambda m: "str(config.get_experiment_path('{}'))".format(m.group(1)),
    r"['\"](classification_report.*?\.txt)['\"]": lambda m: "str(config.get_experiment_path('{}'))".format(m.group(1)),
    r"['\"](evaluation_result.*?\.txt)['\"]": lambda m: "str(config.get_experiment_path('{}'))".format(m.group(1)),
    r"['\"](clean_reuslt\.txt)['\"]": "str(config.get_experiment_path('clean_reuslt.txt'))",
    r"['\"](train_loss_history.*?\.txt)['\"]": lambda m: "str(config.get_experiment_path('{}'))".format(m.group(1)),
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    if "import config" not in content:
        if content.startswith('import ') or content.startswith('from '):
            content = import_block + content
        else:
            match = re.search(r"^(import |from )", content, re.MULTILINE)
            if match:
                idx = match.start()
                content = content[:idx] + import_block + content[idx:]
            else:
                content = import_block + content

    for pat, repl in patterns.items():
        if callable(repl):
            content = re.sub(pat, repl, content)
        else:
            content = re.sub(pat, repl, content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath.name}")

for file in models_dir.glob('**/*.py'):
    process_file(file)

print("Model scripts path refactoring complete.")
