import os
site_pkgs = r'c:\Users\sarda\Downloads\pocus\pocus\backend\venv\Lib\site-packages'
# Print top-level package dirs (skip dist-info etc)
pkgs = []
for name in sorted(os.listdir(site_pkgs)):
    full = os.path.join(site_pkgs, name)
    if os.path.isdir(full) and not name.endswith('.dist-info') and not name.endswith('.data') and not name.startswith('_'):
        pkgs.append(name)
    elif name.endswith('.dist-info'):
        pkgs.append(name.replace('.dist-info',''))
print('\n'.join(sorted(set(pkgs))))
