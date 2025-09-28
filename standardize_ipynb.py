import json
import re
import sys
import os

def title_case(text):
    """Convert text to title case, handling common words."""
    # Words to not capitalize
    lower_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of', 'with', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'within', 'without', 'against', 'along', 'around', 'behind', 'beside', 'besides', 'beyond', 'concerning', 'considering', 'despite', 'except', 'inside', 'like', 'near', 'off', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since', 'throughout', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'upon', 'with', 'within', 'without'}
    
    words = text.split()
    if not words:
        return text
    
    # Capitalize first word
    result = [words[0].capitalize()]
    
    # Handle rest
    for word in words[1:]:
        if word.lower() in lower_words:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    
    return ' '.join(result)

def standardize_markdown(content):
    """Standardize markdown formatting."""
    lines = content.split('\n')
    standardized = []
    
    for line in lines:
        # Standardize headings
        if re.match(r'^#{1,6}\s', line):
            # Extract level and text
            match = re.match(r'^(#{1,6})\s*(.+)$', line)
            if match:
                level, text = match.groups()
                # Title case the heading text
                standardized_text = title_case(text.strip())
                line = f'{level} {standardized_text}'
        
        # Standardize lists
        elif re.match(r'^[\s]*[-\*\+]\s', line):
            # Ensure consistent dash for unordered lists
            line = re.sub(r'^[\s]*[-\*\+]\s', '- ', line)
        
        # Standardize numbered lists
        elif re.match(r'^[\s]*\d+\.\s', line):
            # Ensure proper numbering format
            pass  # Keep as is
        
        # Standardize code blocks
        elif line.strip().startswith('```'):
            # Ensure consistent backticks
            line = '```' + line.strip()[3:]
        
        # Standardize emphasis
        # Ensure consistent bold and italic
        line = re.sub(r'\*\*\*(.+?)\*\*\*', r'**\1**', line)  # ***text*** to **text**
        line = re.sub(r'___(.+?)___', r'**\1**', line)  # ___text___ to **text**
        line = re.sub(r'__(.+?)__', r'**\1**', line)  # __text__ to **text**
        line = re.sub(r'_(.+?)_', r'*\1*', line)  # _text_ to *text*
        
        # Standardize links
        # [text](url) is fine
        
        # Standardize images
        # ![alt](url) is fine
        
        standardized.append(line)
    
    return '\n'.join(standardized)

def standardize_ipynb(file_path):
    """Standardize markdown cells in a Jupyter notebook."""
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if source:
                # Join the source lines
                content = ''.join(source)
                # Standardize
                standardized = standardize_markdown(content)
                # Split back to lines
                cell['source'] = standardized.split('\n')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python standardize_ipynb.py <file.ipynb>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        sys.exit(1)
    
    standardize_ipynb(file_path)
    print(f"Standardized {file_path}")