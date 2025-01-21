import numpy as np
import tqdm


# Function to read the images file and return dictionaries with image IDs, names, and corresponding 3D point IDs
def read_images_file(image_file):
    ''' Extract information from COLMAP image.txt

    Read the COLMAP log and parse data line by line.
    Arg:
        image_file: string, the path to image.txt
    Returns:
        image_id_to_name: dict, a map from image id to image name
        name_to_image_id: dict, a map from image name to image id
        image_id_to_point3d_id: a dictionary of lists, mapping image id to the list of all 3D points' id contained in the view
    '''

    image_id_to_name, name_to_image_id, image_id_to_point3d_id = {}, {}, {}

    with open(image_file, 'r') as f:
        lines = f.readlines()

        # Parse the header to get the number of images
        for i in range(len(lines)):
            if lines[i].find('# Number of images:') > -1:
                num_images = int(lines[i].split(': ')[1].split(',')[0])                
            if not lines[i].startswith('#'):
                line_id_end_header = i
                break
        lines = lines[line_id_end_header:]

        assert (len(lines)/2 == num_images)

        # Parse the rest of the file to get the image IDs, names, and corresponding 3D point IDs
        for i in range(num_images):
            f_line = lines[2*i].strip().split(' ')
            s_line = lines[2*i+1].strip().split(' ')

            image_id = int(f_line[0])
            qw, qx, qy, qz = [float(x) for x in f_line[1:5]]
            tx, ty, tz = [float(x) for x in f_line[5:8]]
            camera_id = int(f_line[8])
            name = f_line[9]
            image_id_to_name[image_id] = name
            name_to_image_id[name] = image_id            
            point3d_id = [int(x) for x in s_line[2::3]]
            point3d_id = np.array(point3d_id, dtype=np.int32)

            image_id_to_point3d_id[image_id] = point3d_id

    return image_id_to_name, name_to_image_id, image_id_to_point3d_id


# Function to compute the number of shared keypoints between two images
def compute_shared_keypoints(image_id_to_point3d_id, image_id1, image_id2):
    ''' Compute the number of intersections of 3D keypoints between two images

    Args:
        image_id_to_point3d_id: a dictionary of lists, the mapping from image id to 3d point id
        image_id1: int, id of the first image
        image_id2: int, id of the second image
    Return:
        The number of intersections
    '''
    point3d_id_1 = image_id_to_point3d_id[image_id1]
    point3d_id_2 = image_id_to_point3d_id[image_id2]        
    point3d_id_1 = point3d_id_1[point3d_id_1 != -1]
    point3d_id_2 = point3d_id_2[point3d_id_2 != -1]
    common_point3d_id = np.intersect1d(point3d_id_1, point3d_id_2)

    return len(common_point3d_id)

# Function to compute the distance matrix between all pairs of images
def compute_distance_matrix(image_id_to_point3d_id, image_ids):
    ''' Constructing photogrammetric distance metric

    Args:
        image_id_to_point3d_id: a dictionary of lists, the mapping from image id to 3d point id
        image_ids: list
    Return:
        D: numpy matrix of all_views_num x all_views_num
    '''

    D = np.zeros((len(image_ids), len(image_ids)), dtype=np.int32)
    n_total = len(image_ids) * (len(image_ids))
    with tqdm.tqdm(total=n_total,desc='Computing shared keypoints') as pbar:
        for i in range(len(image_ids)):
           for j in range(i, len(image_ids)):
                pbar.update(1)
                if i == j:
                    continue
                else:
                    image_id1 = image_ids[i]
                    image_id2 = image_ids[j]
                    shared_keypoints = compute_shared_keypoints(image_id_to_point3d_id, image_id1, image_id2)
                    D[i, j] = shared_keypoints
                    D[j, i] = shared_keypoints

    # This distance matrix should be symmetric, since D_ij = D_ji
    if not np.allclose(D, D.T):
        raise ValueError('The distance matrix is not symmetric!')

    return D

def get_3d_correspondence_matrix(colmap_image_log, image_names):
    ''' 3D keypoints matching
    
    Arg:
        colmap_image_log: string, filepath of colmap generated 'image.txt'
        image_names: list, list of views file path
    Return:
        D: numpy matrix of all_views_num x all_views_num
    '''
    
    _, name_to_image_id, image_id_to_point3d_id = read_images_file(colmap_image_log)
    image_ids = [ name_to_image_id[x] for x in image_names ]

    D = compute_distance_matrix(image_id_to_point3d_id, image_ids)
    D = 1. - D / np.max(D)
    
    return D