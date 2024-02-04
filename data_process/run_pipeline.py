import sys
import os 
# Generate file list first

task_id = str(int(sys.argv[1]))
lstin = './filename_lists/name_list_%s.txt'%(task_id)


# Multi-Person Detect and get the bounding boxes
print('############# Face detect start #############')
detres_dir = 'det_res'
if not os.path.exists(detres_dir):
   os.makedirs(detres_dir)
os.system('python detectface.py %s %s %s'%(lstin, os.path.join(detres_dir, 'det_res_%s.txt'%(task_id)), os.path.join(detres_dir, 'det_multiface_%s.txt'%(task_id))))

# Seg Face 
print('############# Face segmentation start #############')
os.system('python face_seg.py %s'%(os.path.join(detres_dir, 'det_res_%s.txt')%(task_id)))

# crop and align
print('############# Face crop and align start #############')
os.system('python crop_align_face.py --input-lst %s'%(os.path.join(detres_dir, 'det_res_%s.txt')%(task_id)))
next_file = os.path.join(detres_dir, 'det_res_%s.txt'%(task_id))


# # Hand detect
# handet_dir = 'handet_dir'
# next_file = './handet_dir/no_hand_%s.txt'%(task_id)
# print('############# Hand detect start #############')
# if not os.path.exists(handet_dir):
#     os.makedirs(handet_dir)
# os.system('python hand_det.py %s %s %s'%(os.path.join(detres_dir, 'det_res_%s.txt'%(task_id)), os.path.join(handet_dir, 'no_hand_%s.txt'%(task_id)), os.path.join(handet_dir, 'with_hand_%s.txt'%(task_id))))

# Face Quality Estimate on Aligned
print('############# Face Quality Select start #############')
quality_dir = 'quality_dir'
if not os.path.exists(quality_dir):
    os.makedirs(quality_dir)
os.system('python face_quality.py --input_lst=%s --output_file=%s --low_output_file=%s'%(next_file, os.path.join(quality_dir, 'valid_data_%s.txt'%(task_id)), os.path.join(quality_dir, 'quality_low_%s.txt'%(task_id))) )

# BLIP caption
crop_dir = 'cropimg'
caption_dir = './caption_res/caption_res_%s'%(task_id)
if not os.path.exists(caption_dir):
    os.makedirs(caption_dir)
os.system('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python caption.py %s %s %s'%('./quality_dir/valid_data_%s.txt'%(task_id), crop_dir, caption_dir))

# create json for training
os.system('python create_json.py %s'%('./caption_res/caption_res_%s'%(task_id)))




