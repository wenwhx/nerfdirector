import numpy as np


def great_circle_dist(p1, p2):
    ''' Caluclate the great-circle distance between reference point and others

    Args:
    p1: numpy array with shape [d,], reference point
    p2: numpy array with shape [N, d,], target points
    is_spherical: bool
    '''
    chord = np.sqrt(((p1 - p2) ** 2).sum(-1))
    
    chord = np.clip(chord, -2.0, 2.0)
    return 2 * np.arcsin(chord / 2.0)

def euclidean_dist(p1, p2):
    ''' Compute the Euclidean distance between reference point and others in cartessian coordinate
    
    Args:
        p1: numpy array with the shape [3]
        p2: numpy array with the shape [N, 3]
    '''
    
    p1 = np.reshape(p1, (1, 3))
    dist = np.sum( (p2 - p1) ** 2, axis=1 )
    
    return dist

def farthest_view_sampling(K, candidates, seed, dist_type='euc', selected_status=[]):
    ''' Farthest view sampling according to the distance between camera centers

    Args:
        K: int, number of views to be selected
        candidates: list, list of all candidate camera centers
        seed: int, random seed
        dist_type: str, specify the spatial distance metric, including great circle distance and Euclidean distance
        selected_status: list, indicate if one view is selected or not
    Return:
        selected_points: list, all selected camera centers
    '''
    dist_dict = {
        'gcd' : 'great_circle_dist',
        'euc' : 'euclidean_dist'
    }
    
    np.random.seed(seed)

    # randomly select N points into the waiting list
    points = np.array(candidates)
    if dist_type == 'gcd':
        radius = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / radius

    # initialize dist, point_left_idx and selected_points
    N = points.shape[0]
    dist = np.full(N, np.inf)
    point_left_idx = np.arange(N)
    selected_points = []
    
    # initialize distance function
    dist_func = dist_dict[dist_type] if dist_type in dist_dict.keys() else 'great_circle_dist'
    print('current dist func is {}'.format(dist_func))

    # initialize dist list if selected_index provided
    # else random sample the first active point
    if len(selected_status) > 0:
        selected_points = point_left_idx[selected_status].tolist()
        point_left_idx = np.delete(point_left_idx, selected_points)
        for index in selected_points:
            p = points[index, :]
            dist_to_active_point = globals()[dist_func](p, points[point_left_idx])
            dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])

        selected_index = selected_points[-1]
        start = 0

    else:
        # sample first active point
        selected_index = np.random.randint(0, N - 1)
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, selected_index)
        start = 1
        
    for i in range(start, K):
        active_point = points[selected_index, :]

        # get the distance from points in waiting list to the active point
        dist_to_active_point = globals()[dist_func](active_point, points[point_left_idx])

        # find the nearest neighbor in the selected list for each point in the waiting list
        dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])
        
        # find the farthest nearest neighbor
        selected_index = point_left_idx[np.argmax(dist[point_left_idx])]
        
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, np.argmax(dist[point_left_idx]))


    return selected_points

def farthest_view_sampling_colmap(K, candidates, seed, D, dist_type='euc', selected_status=[]):
    ''' Farthest view sampling according to both spatial and photogrammetric distance

    Args:
        K: int, number of views to be selected
        candidates: list, list of all candidate camera centers
        seed: int, random seed
        D: numpy matrix of all_views_num x all_views_num
        dist_type: str, specify the spatial distance metric, including great circle distance and Euclidean distance
        selected_status: list, indicate if one view is selected or not
    Return:
        selected_points: list, all selected camera centers
    '''
    dist_dict = {
        'gcd' : 'great_circle_dist',
        'euc' : 'euclidean_dist'
    }
    
    np.random.seed(seed)

    # randomly select N points into the waiting list
    points = np.array(candidates)
    d = D
    if dist_type == 'gcd':
        radius = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / radius
        d = D * np.pi # gcd's range: [0, pi]
        print(d)
    elif dist_type == "euc":
        radius_list = np.linalg.norm(points, axis=1, keepdims=True)
        d = D * np.max(radius_list) * 2
        print(d)

    # initialize dist, point_left_idx and selected_points
    N = points.shape[0]
    dist = np.full(N, np.inf)
    point_left_idx = np.arange(N)
    selected_points = []
    
    # initialize distance function
    dist_func = dist_dict[dist_type] if dist_type in dist_dict.keys() else 'great_circle_dist'
    print('current dist func is {}'.format(dist_func))

    # initialize dist list if selected_index provided
    # else random sample the first active point
    if len(selected_status) > 0:
        selected_points = point_left_idx[selected_status].tolist()
        point_left_idx = np.delete(point_left_idx, selected_points)
        for index in selected_points:
            p = points[index, :]
            dist_to_active_point = globals()[dist_func](p, points[point_left_idx])
            
            # fetch 3d correspondence distance
            print("1.. Fetching [{}, ] from D...".format(index))
            dist_3d_to_active_point = d[index, point_left_idx]
            dist_to_active_point += dist_3d_to_active_point
            dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])
        selected_index = selected_points[-1]
        start = 0
    else:
        # sample first active point
        selected_index = np.random.randint(0, N - 1)
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, selected_index)
        start = 1
        
    for i in range(start, K):
        active_point = points[selected_index, :]

        # get the distance from points in waiting list to the active point
        dist_to_active_point = globals()[dist_func](active_point, points[point_left_idx])
        
        # fetch 3d correspondence distance
        print("2.. Fetching [{}, ] from D...".format(selected_index))
        dist_3d_to_active_point = d[selected_index, point_left_idx]
        dist_to_active_point += dist_3d_to_active_point

        dist[point_left_idx] = np.minimum(dist_to_active_point, dist[point_left_idx])
        
        # find the neighbor satisfying: 1) the farthest nearest, and 2) the most different
        selected_index = point_left_idx[np.argmax(dist[point_left_idx])]
        
        selected_points.append(selected_index)
        point_left_idx = np.delete(point_left_idx, np.argmax(dist[point_left_idx]))


    return selected_points