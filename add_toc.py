import re
import sys

file = sys.argv[1]

with open(file, "r") as f:
    content = f.read()

# Find all headings
headings = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)

toc = "# Table of Contents\n\n"
for level, title in headings:
    indent = "  " * (len(level) - 1)
    link = re.sub(r"[^\w\s-]", "", title).lower().replace(" ", "-")
    toc += f"{indent}- [{title}](#{link})\n"

# Insert after the first # heading
lines = content.split("\n")
insert_pos = 0
for i, line in enumerate(lines):
    if line.startswith("# "):
        insert_pos = i + 1
        break

lines.insert(insert_pos, "")
lines.insert(insert_pos, toc)

new_content = "\n".join(lines)

with open(file, "w") as f:
    f.write(new_content)
