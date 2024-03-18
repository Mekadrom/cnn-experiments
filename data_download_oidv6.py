import os

with open('classes.txt', 'r') as f:
    classes = f.read().splitlines()

# shell command `oidv6 downloader --dataset data/ --type_data all --classes {}`
    
os.system('oidv6 downloader --dataset data/ --type_data all --classes {} --yes --multi_classes --limit 10'.format(' '.join(['"' + clss + '"' for clss in classes])))
