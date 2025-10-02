#!/usr/bin/env python3
"""
Delete all .bak backup files from the CAs directory
"""

import os
from pathlib import Path

def delete_bak_files(directory):
    """Delete all .bak files in directory and subdirectories."""
    deleted_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bak'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"✓ Deleted: {os.path.relpath(file_path, directory)}")
                except Exception as e:
                    print(f"✗ Failed to delete {file_path}: {e}")
    
    return deleted_files

def main():
    cas_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Backup File Cleaner (.bak)")
    print("=" * 70)
    print(f"\nSearching for .bak files in: {cas_dir}\n")
    
    deleted = delete_bak_files(str(cas_dir))
    
    print("\n" + "=" * 70)
    print(f"✓ Deleted {len(deleted)} backup file(s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
