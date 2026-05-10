import ast
import os

def get_hardcoded_paths_and_execution(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return {'error': str(e)}

    has_main = False
    has_top_level_execution = False
    hardcoded_paths = set()
    imports = []
    
    try:
        tree = ast.parse(source)
    except Exception as e:
        return {'error': 'Parse error: ' + str(e)}

    # Check for execution blocks and hardcoded paths
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check for if __name__ == '__main__':
            try:
                if (isinstance(node.test, ast.Compare) and 
                    isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and
                    isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == '__main__'):
                    has_main = True
            except:
                pass
        
        # Check for paths in strings
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if any(ext in val for ext in ['.csv', '.png', '.jpg', '.jpeg', '.pth', '.pt', '.txt', 'C:/Users/', 'C:\\Users\\', '.h5', '.hdf5']):
                hardcoded_paths.add(val)

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        
    # check top level nodes
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            has_top_level_execution = True
        if isinstance(node, ast.Assign):
            # Only consider assignment as execution if it's a call
            if isinstance(node.value, ast.Call):
                has_top_level_execution = True

    return {
        'has_main': has_main,
        'has_top_level_execution': has_top_level_execution,
        'paths': list(hardcoded_paths),
        'imports': list(set(imports))
    }

results = {}
for root, _, files in os.walk('.'):
    # skip .git, env, __pycache__, etc.
    if any(ignore in root for ignore in ['.git', '__pycache__', '.ipynb_checkpoints']):
        continue

    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            results[path] = get_hardcoded_paths_and_execution(path)

for path, info in results.items():
    print(f"--- {path} ---")
    if 'error' in info:
        print(f"Error: {info['error']}")
    else:
        print(f"Has if __name__ == '__main__': {info['has_main']}")
        print(f"Has top-level execution logic: {info['has_top_level_execution']}")
        if info['paths']:
            print("Hardcoded Paths:")
            for p in info['paths']:
                print(f"  - {p}")
        else:
            print("Hardcoded Paths: None")
    print()
