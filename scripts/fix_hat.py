"""Replace \\hat with \\widehat in all docs .md files."""
import os

docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')

total_files = 0
total_replacements = 0

for root, dirs, files in os.walk(docs_dir):
    for fname in files:
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath, encoding='utf-8') as f:
            text = f.read()
        # \hat is safe to replace: it is NOT a substring of \widehat
        # (\widehat starts with \w, not \h)
        count = text.count('\\hat')
        if count > 0:
            new_text = text.replace('\\hat', '\\widehat')
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_text)
            total_files += 1
            total_replacements += count
            print(f"  {fname}: {count} replacements")

print(f"\nDone: {total_replacements} replacements across {total_files} files")
