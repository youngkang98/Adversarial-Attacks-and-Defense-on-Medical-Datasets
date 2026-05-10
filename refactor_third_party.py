import os
import re
from pathlib import Path

third_party_dir = Path('third_party')

import_block = """import sys
from pathlib import Path
# Dynamically find the project root (where config.py lives)
current_dir = Path(__file__).resolve().parent
project_root = current_dir
while not (project_root / 'config.py').exists() and project_root != project_root.parent:
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
try:
    import config
except ImportError:
    pass # If config isn't found, we don't break the script immediately

"""

# Regex patterns to find hardcoded paths in third_party
patterns = {
    # TXT split files
    r"['\"]C:/Users/[^'\"]+/txt/test0\.txt['\"]": "str(config.get_data_path('train0.txt')).replace('train0.txt', 'test0.txt')", # Assuming test0 exists or will
    r"['\"]C:/Users/[^'\"]+/txt/train0\.txt['\"]": "str(config.get_data_path('train0.txt'))",
    r"['\"]C:/Users/[^'\"]+/txt/validation/?['\"]": "str(config.get_data_path('validation'))",
    r"['\"]C:/Users/[^'\"]+/txt/ISIC2019_grandtruethlabels\.csv['\"]": "str(config.get_data_path('ISIC2019_labels.csv'))",
    
    # Datasets - ISIC / CSV
    r"['\"]C:/Users/[^'\"]+/New UAP/(ISIC2019_.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/(ISIC2019_.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/New UAP/(CXRAY-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/New UAP/(OCT2017-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/GitHub/Adversarial-Attack-and-Defense/(CXRAY-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    r"['\"]C:/Users/[^'\"]+/Github/Adversarial-Attack-and-Defense/(CXRAY-.*?\.csv)['\"]": lambda m: "str(config.get_data_path('{}'))".format(m.group(1)),
    
    # Direct CSV names
    r"['\"](data/)?OCT2017-test\.csv['\"]": "str(config.get_data_path('OCT2017-test.csv'))",
    r"['\"](data/)?OCT2017-train\.csv['\"]": "str(config.get_data_path('OCT2017-train.csv'))",
    r"['\"](data/)?OCT2017-train-adv-0\.1\.csv['\"]": "str(config.get_data_path('OCT2017-train-adv-0.1.csv'))",
    
    # Image Directories
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_Training_Input/?(?:ISIC_2019_Training_Input/)?['\"]": "str(config.get_data_path('ISIC2019'))",
    r"['\"]C:/Users/[^'\"]+/ISIC_2019_Test_Input/?['\"]": "str(config.get_data_path('ISIC2019_test'))",
    
    # Echocardiogram
    r"['\"]C:/Users/[^'\"]+/Echocardiogram/Echocardiogram/data_split/train/?['\"]": "str(config.get_data_path('Echocardiogram/train'))",
    r"['\"]C:/Users/[^'\"]+/Echocardiogram/Echocardiogram/data_split/test/?['\"]": "str(config.get_data_path('Echocardiogram/test'))",

    # COVID / Chest X-ray
    r"['\"]C:/Users/[^'\"]+/COVID-Net/?['\"]": "str(config.get_data_path('COVID-Net'))",
    r"['\"]C:/Users/[^'\"]+/COVID-Net/test/?['\"]": "str(config.get_data_path('COVID-Net/test'))",
    r"['\"]C:\\Users\\[^'\"]+\\COVID-Net/?['\"]": "str(config.get_data_path('COVID-Net'))",
    r"['\"]C:/Users/[^'\"]+/chest_xray/train/?['\"]": "str(config.get_data_path('chest_xray/train'))",
    r"['\"]C:/Users/[^'\"]+/chest_xray/test/?['\"]": "str(config.get_data_path('chest_xray/test'))",
    
    # specific wanet/uap splits
    r"['\"](\.\./COVID-Net/train_split_v3\.txt)['\"]": "str(config.get_data_path('train_split_v3.txt'))",
    r"['\"](\.\./COVID-Net/test_split_v3\.txt)['\"]": "str(config.get_data_path('test_split_v3.txt'))",
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    has_target = any(re.search(pat, content) for pat in patterns.keys())

    if has_target:
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
        print(f"Updated {filepath}")

for file in third_party_dir.glob('**/*.py'):
    process_file(file)

print("Third-party scripts path refactoring complete.")
