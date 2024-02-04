import os
import json
import sys

src = sys.argv[1]
# src = 'caption_res'
caption_dir = 'caption'
if not os.path.exists(caption_dir):
    os.makedirs(caption_dir)
Files = os.listdir(src)
idmap = {}
for file in sorted(Files):
    txtpath = os.path.join(src, file)
    txtfile = open(txtpath, 'r')
    txt = txtfile.read().strip()
    fid = file.split('_')[0]
    if not fid in idmap:
        idmap[fid] = {'imgs': [], 'captions': []}
    idmap[fid]['imgs'].append(file.rsplit('.')[0]+'.jpg')
    idmap[fid]['captions'].append(txt)

for fid in idmap:
    if len(idmap[fid]['imgs']) > 2:
        caption_out = open(os.path.join(caption_dir, fid+'.json'), 'w')
        caption_out.write(json.dumps(idmap[fid]))
        print(caption_out, 'has been captioned')
