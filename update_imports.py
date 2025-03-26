#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
from pathlib import Path

# The flexible import code to add to each page
IMPORT_CODE = """
# Use flexible import approach for the api module
try:
    # Try first as absolute import from app structure
    from app.streamlit_app.api import *
except ImportError:
    try:
        # Try as relative import
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import API module
        from api import *
    except ImportError as e:
        st.error(f"Error importing API module: {e}")
"""

def update_page_imports(file_path):
    """Update the imports in a streamlit page file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already has the new import
    if "flexible import approach" in content:
        print(f"File {file_path} already updated, skipping...")
        return
    
    # Find import section and replace old one with new one
    import_pattern = r"# Use relative import for the api module.*?(?=\ndef show\(\)|$)"
    import_pattern_flags = re.DOTALL | re.MULTILINE
    
    if re.search(import_pattern, content, import_pattern_flags):
        new_content = re.sub(import_pattern, IMPORT_CODE, content, flags=import_pattern_flags)
    else:
        # If pattern not found, add after imports
        lines = content.split('\n')
        import_lines = []
        non_import_lines = []
        
        # Find where imports end
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            else:
                non_import_lines.append(line)
        
        # Combine with new import code
        new_content = '\n'.join(import_lines) + IMPORT_CODE + '\n'.join(non_import_lines)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated imports in {file_path}")

def main():
    # Get the path to the pages directory
    pages_dir = Path(__file__).parent / 'app' / 'streamlit_app' / 'pages'
    
    if not pages_dir.exists():
        print(f"Pages directory not found at {pages_dir}")
        return 1
    
    # Process all Python files in the pages directory
    for file_path in pages_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        
        print(f"Processing {file_path}...")
        update_page_imports(file_path)
    
    print("\nAll page imports updated successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 