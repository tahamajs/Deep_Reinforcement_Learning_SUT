#!/usr/bin/env python3
"""
Jupyter Notebook Comment Cleaner
Removes all comments from code cells in .ipynb files while preserving code functionality.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import shutil


def remove_python_comments(code: str) -> str:
    """
    Remove Python comments from code while preserving strings.
    
    Args:
        code: Python code as string
        
    Returns:
        Code with comments removed
    """
    lines = code.split('\n')
    cleaned_lines = []
    in_multiline_string = False
    multiline_delimiter = None
    
    for line in lines:
        if not in_multiline_string:
            stripped = line.lstrip()
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                delimiter = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(delimiter) == 2 and len(stripped) > 6:
                    continue
                elif stripped.count(delimiter) == 1:
                    in_multiline_string = True
                    multiline_delimiter = delimiter
                    continue
            
            if '#' in line:
                in_string = False
                string_char = None
                escape_next = False
                cleaned_line = []
                
                for i, char in enumerate(line):
                    if escape_next:
                        cleaned_line.append(char)
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        cleaned_line.append(char)
                        escape_next = True
                        continue
                    
                    if char in ['"', "'"]:
                        if not in_string:
                            in_string = True
                            string_char = char
                            cleaned_line.append(char)
                        elif char == string_char:
                            in_string = False
                            string_char = None
                            cleaned_line.append(char)
                        else:
                            cleaned_line.append(char)
                        continue
                    
                    if char == '#' and not in_string:
                        break
                    
                    cleaned_line.append(char)
                
                line = ''.join(cleaned_line).rstrip()
            
            if line.strip():
                cleaned_lines.append(line)
        else:
            if multiline_delimiter in line:
                in_multiline_string = False
                multiline_delimiter = None
    
    return '\n'.join(cleaned_lines)


def clean_notebook_comments(notebook_path: str, create_backup: bool = True) -> Dict[str, Any]:
    """
    Remove comments from all code cells in a Jupyter notebook.
    
    Args:
        notebook_path: Path to the .ipynb file
        create_backup: Whether to create a backup file
        
    Returns:
        Dictionary with statistics about the cleaning operation
    """
    stats = {
        'original_lines': 0,
        'cleaned_lines': 0,
        'cells_processed': 0,
        'comments_removed': 0
    }
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if create_backup:
            backup_path = notebook_path + '.bak'
            shutil.copy2(notebook_path, backup_path)
            print(f"✓ Backup created: {backup_path}")
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                
                if isinstance(source, list):
                    original_code = ''.join(source)
                else:
                    original_code = source
                
                original_lines = len(original_code.split('\n'))
                stats['original_lines'] += original_lines
                
                cleaned_code = remove_python_comments(original_code)
                
                cleaned_lines = len(cleaned_code.split('\n'))
                stats['cleaned_lines'] += cleaned_lines
                stats['comments_removed'] += (original_lines - cleaned_lines)
                
                if isinstance(source, list):
                    cell['source'] = [cleaned_code + '\n'] if cleaned_code else []
                else:
                    cell['source'] = cleaned_code
                
                stats['cells_processed'] += 1
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        return stats
        
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {str(e)}")
        return stats


def find_notebooks(directory: str) -> List[str]:
    """Find all .ipynb files in directory and subdirectories."""
    notebooks = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb') and not file.endswith('.ipynb.bak'):
                if '.ipynb_checkpoints' not in root:
                    notebooks.append(os.path.join(root, file))
    return notebooks


def main():
    """Main function to clean comments from all notebooks in CAs directory."""
    cas_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Jupyter Notebook Comment Cleaner")
    print("=" * 70)
    print(f"\nSearching for notebooks in: {cas_dir}")
    
    notebooks = find_notebooks(str(cas_dir))
    
    if not notebooks:
        print("No notebook files found!")
        return
    
    print(f"\nFound {len(notebooks)} notebook(s)")
    print("\nDo you want to create backups? (y/n): ", end='')
    
    create_backup = input().strip().lower() == 'y'
    
    print(f"\nProcessing notebooks{'with backups' if create_backup else 'WITHOUT backups'}...\n")
    
    total_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_original_lines': 0,
        'total_cleaned_lines': 0,
        'total_comments_removed': 0,
        'total_cells_processed': 0
    }
    
    for i, notebook_path in enumerate(notebooks, 1):
        rel_path = os.path.relpath(notebook_path, cas_dir)
        print(f"[{i}/{len(notebooks)}] Processing: {rel_path}")
        
        stats = clean_notebook_comments(notebook_path, create_backup)
        
        if stats['cells_processed'] > 0:
            total_stats['files_processed'] += 1
            total_stats['total_original_lines'] += stats['original_lines']
            total_stats['total_cleaned_lines'] += stats['cleaned_lines']
            total_stats['total_comments_removed'] += stats['comments_removed']
            total_stats['total_cells_processed'] += stats['cells_processed']
            
            print(f"  ✓ Processed {stats['cells_processed']} code cell(s)")
            print(f"  ✓ Removed {stats['comments_removed']} line(s) of comments")
        else:
            total_stats['files_failed'] += 1
            print(f"  ✗ Failed or no code cells")
        
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed successfully: {total_stats['files_processed']}")
    print(f"Files failed: {total_stats['files_failed']}")
    print(f"Total code cells processed: {total_stats['total_cells_processed']}")
    print(f"Original lines: {total_stats['total_original_lines']}")
    print(f"Cleaned lines: {total_stats['total_cleaned_lines']}")
    print(f"Comments removed: {total_stats['total_comments_removed']}")
    print(f"Reduction: {total_stats['total_comments_removed']} lines ({100 * total_stats['total_comments_removed'] / max(total_stats['total_original_lines'], 1):.1f}%)")
    print("=" * 70)
    print("\n✓ All notebooks processed!")
    
    if create_backup:
        print("\n💡 Tip: Backup files (.bak) have been created. You can restore them if needed.")


if __name__ == "__main__":
    main()
