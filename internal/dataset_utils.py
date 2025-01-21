import json
import os

import numpy as np

def get_camera_centers(transform_json_path):
    with open(transform_json_path, 'r') as f:
        data = json.load(f) 

    camera_centers = []
    for frame in data['frames']:
        camera_to_world = np.array(frame['transform_matrix'], dtype=np.float32)
        camera_center = camera_to_world[:3, 3]
        
        camera_centers += [camera_center[None].copy()]
    camera_centers = np.array(camera_centers)

    print('Load {} cameras from {}'.format(camera_centers.shape[0], transform_json_path))

    return np.squeeze(camera_centers)

def generate_new_transform(base_transform, target_transform, all_train_transform, new_frame_idxs=[]):
    # set the relative path of transforms.json to images
    with open(all_train_transform, 'r') as f:
        pools = json.load(f)
        all_frames = pools['frames']
    src_img = os.path.join(os.path.dirname(all_train_transform), all_frames[0]['file_path'])
    rel_path = os.path.relpath(src_img, os.path.dirname(target_transform))
    rel_path = os.path.dirname(rel_path)
    
    # load base_transform information if base transform exists
    # otherwise, just load the head info of all transform
    if os.path.exists(base_transform):
        with open(base_transform, 'r') as f:
            data = json.load(f)
    else:
        data = pools.copy()
        data['frames'] = []

    if len(new_frame_idxs) > 0:
        # correct relative path and add new the newly selected views
        for i in new_frame_idxs:
            f = all_frames[i]
            filename = os.path.basename(f['file_path'])
            f['file_path'] = os.path.join(rel_path, filename)
            data['frames'].append(f)
    else:
        # set correct rel_path to base_transform frames
        for f in data['frames']:
            filename = os.path.basename(f['file_path'])
            f['file_path'] = os.path.join(rel_path, filename)
    
    with open(target_transform, 'w') as f:
        json.dump(data, f, indent=4)