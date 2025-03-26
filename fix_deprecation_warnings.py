#!/usr/bin/env python
"""
Fix Deprecation Warnings

This script finds and fixes deprecated usage in the codebase.
Current fixes:
1. Pandas date_range 'H' frequency to 'h'
2. Styler.applymap to Styler.map
3. use_column_width to use_container_width
"""

import os
import sys
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("fix_deprecation_warnings")

def fix_date_range_frequency(content: str) -> str:
    """Fix deprecated frequency parameters in pd.date_range calls."""
    # Pattern 1: freq='1H' or freq="1H"
    pattern1 = r"freq=['\"](\d+)H['\"]"
    # Pattern 2: freq='H' or freq="H"
    pattern2 = r"freq=['\"]H['\"]"
    # Pattern 3: freq='1D' or freq="1D"
    pattern3 = r"freq=['\"](\d+)D['\"]"
    
    def replace_freq1(match):
        return f"freq='{match.group(1)}h'"
    
    def replace_freq2(match):
        return "freq='h'"
    
    def replace_freq3(match):
        return f"freq='{match.group(1)}d'"
    
    # Apply all replacements
    content = re.sub(pattern1, replace_freq1, content)
    content = re.sub(pattern2, replace_freq2, content)
    content = re.sub(pattern3, replace_freq3, content)
    
    return content

def fix_styler_applymap(content):
    """
    Fix deprecated Styler.applymap to Styler.map
    """
    # Match dataframe.style.applymap
    pattern = r"\.style\.applymap\("
    replacement = ".style.map("
    
    # Replace the pattern
    fixed_content = re.sub(pattern, replacement, content)
    
    return fixed_content

def fix_use_column_width(content):
    """
    Fix deprecated use_column_width to use_container_width
    """
    # Match use_column_width
    pattern = r"use_column_width\s*=\s*(True|False)"
    
    def replace_param(match):
        value = match.group(1)
        return f"use_container_width={value}"
    
    # Replace the pattern
    fixed_content = re.sub(pattern, replace_param, content)
    
    return fixed_content

def fix_file(file_path: str) -> bool:
    """Fix deprecation warnings in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for potential issues
        has_warnings = False
        if 'freq=' in content:
            has_warnings = True
        
        if has_warnings:
            fixed_content = fix_date_range_frequency(content)
            if fixed_content != content:
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                logger.info(f"Fixed deprecation warnings in {file_path}")
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    """Main function to fix deprecation warnings in files."""
    if len(sys.argv) < 2:
        logger.error("Please provide a file path or directory")
        sys.exit(1)
    
    target = sys.argv[1]
    files_fixed = 0
    
    if os.path.isfile(target):
        if fix_file(target):
            files_fixed += 1
    elif os.path.isdir(target):
        for root, _, files in os.walk(target):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if fix_file(file_path):
                        files_fixed += 1
    else:
        logger.error(f"Target {target} does not exist")
        sys.exit(1)
    
    logger.info(f"Fixed deprecation warnings in {files_fixed} file(s)")

if __name__ == "__main__":
    main() 