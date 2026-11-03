import os, glob
base = r"c:\Users\sarda\Downloads\pocus\pocus"
# Walk and find first .avi
for root, dirs, files in os.walk(base):
    for f in files:
        if f.lower().endswith('.avi'):
            print(os.path.join(root, f))
            break
    else:
        continue
    break

# Also show dataset dir tree top 2 levels
for root, dirs, files in os.walk(os.path.join(base, "dataset")):
    level = root.replace(base, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    if level >= 2:
        dirs[:] = []  # stop recursing deeper
