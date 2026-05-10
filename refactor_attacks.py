import os
import re
from pathlib import Path

attacks_dir = Path('src/attacks')

import_block = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config

"""

# Regex patterns to find hardcoded paths
# We look for strings starting with 'C:/Users/' or 'Wanet/' or specific files
patterns = {
    # Models
    r"['\"](?:\.\./[^'\"]+|C:/Users/[^'\"]+)/.*?(\w+\.pth(?:\.tar)?)['\"]": lambda m: f"str(config.get_model_path('{m.group(1)}'))",
    
    # Checkpoints
    r"['\"]C:/Users/[^'\"]+/checkpoint\.pth['\"]": "str(config.get_model_path('checkpoint.pth'))",
    r"['\"]C:/Users/[^'\"]+/classifier_checkpoint\.pth['\"]": "str(config.get_model_path('classifier_checkpoint.pth'))",
    r"['\"]epoch10_model\.pth['\"]": "str(config.get_model_path('epoch10_model.pth'))",
    
    # Datasets - ISIC / CSV
    r"['\"]C:/Users/[^'\"]+/New UAP/(ISIC2019_.*?\.csv)['\"]": lambda m: f"str(config.get_data_path('{m.group(1)}'))",
    r"['\"]C:/Users/[^'\"]+/New UAP/(CXRAY-.*?\.csv)['\"]": lambda m: f"str(config.get_data_path('{m.group(1)}'))",
    r"['\"]C:/Users/[^'\"]+/New UAP/(OCT2017-.*?\.csv)['\"]": lambda m: f"str(config.get_data_path('{m.group(1)}'))",
    
    # Direct CSV names in the same folder or with no path
    r"['\"]CXRAY-test\.csv['\"]": "str(config.get_data_path('CXRAY-test.csv'))",
    r"['\"]CXRAY-train\.csv['\"]": "str(config.get_data_path('CXRAY-train.csv'))",
    r"['\"]OCT2017-test\.csv['\"]": "str(config.get_data_path('OCT2017-test.csv'))",
    r"['\"]OCT2017-train-adv-0\.1\.csv['\"]": "str(config.get_data_path('OCT2017-train-adv-0.1.csv'))",
    
    # ISIC Image Directories
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_Training_Input/?(?:ISIC_2019_Training_Input/)?['\"]": "str(config.get_data_path('ISIC2019'))",
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_train/?['\"]": "str(config.get_data_path('ISIC2019'))",
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_test/?['\"]": "str(config.get_data_path('ISIC2019'))",
    
    # COVID / Chest X-ray
    r"['\"]C:/Users/[^'\"]+/COVID-Net/?['\"]": "str(config.get_data_path('COVID-Net'))",
    r"['\"]Wanet/(train_COVIDx8B\.txt)['\"]": lambda m: f"str(config.get_data_path('{m.group(1)}'))",
    r"['\"]Wanet/(test_COVIDx8B\.txt)['\"]": lambda m: f"str(config.get_data_path('{m.group(1)}'))",
    
    # OCT Directories
    r"['\"]C:/Users/[^'\"]+/archive/OCT2017/?['\"]": "str(config.get_data_path('OCT'))",
    
    # Results / Output files - using regex for savefig
    r"plt\.savefig\s*\(\s*f?['\"]([^'\"]+?\.png)['\"]": r"plt.savefig(str(config.get_experiment_path(f'\1'))",
    
    # Classification reports and eval results
    r"['\"](classification_report.*?\.txt)['\"]": lambda m: f"str(config.get_experiment_path('{m.group(1)}'))",
    r"['\"](evaluation_result.*?\.txt)['\"]": lambda m: f"str(config.get_experiment_path('{m.group(1)}'))",
    r"['\"](clean_reuslt\.txt)['\"]": "str(config.get_experiment_path('clean_reuslt.txt'))",
    
    # Specific f-strings for outputs (e.g. f'{resultFolder}evaluation_results_{adv_image_count}...')
    r"f?['\"](\{result_?Folder\}[^'\"]+\.txt)['\"]": lambda m: "str(config.get_experiment_path(f'{}'))".format(m.group(1).replace("{resultFolder}", "").replace("{result_folder}", "")),
    
    # Specific paths for adversarial images
    r"f['\"](ISIC2019/adv_img.*?\.jpg)['\"]": lambda m: "str(config.get_experiment_path(f'{}'))".format(m.group(1).replace("ISIC2019/", "")),
    r"f['\"](ISIC2019/psnr.*?\.txt)['\"]": lambda m: "str(config.get_experiment_path(f'{}'))".format(m.group(1).replace("ISIC2019/", "")),
    r"f['\"](\{resultFolder\}(?:adv_img|psnr).*?\.(?:jpg|txt))['\"]": lambda m: "str(config.get_experiment_path(f'{}'))".format(m.group(1).replace("{resultFolder}", ""))
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    if "import config" not in content:
        # Find a good place to insert import_block (after standard imports)
        # Just insert at the top after initial docstrings/shebang if any, or just at line 0
        if content.startswith('import ') or content.startswith('from '):
            content = import_block + content
        else:
            # try to find first import
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

for file in attacks_dir.glob('**/*.py'):
    process_file(file)

print("Attack scripts path refactoring complete.")
