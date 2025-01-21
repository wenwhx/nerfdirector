import numpy as np

from internal.lloyd import lloyd_relaxation


def zipf_sampling(K, eval_values, cur_centers, val_centers, metrics, top=False, lloyd=True, strict=False, crop=False):
    eval_values = np.array(eval_values)
    
    if metrics == 'mse' or metrics == "uncertainty":
        eval_values = -eval_values
    ranking_ind = np.argsort(eval_values)

    # strict greedy: select top K points with highest ranking
    if strict:
        sampled_new_points = val_centers[ranking_ind[:K]]
        print(eval_values[ranking_ind[:K]])

    # use ranking as probability distribution and based on this randomly selected K points
    else:
        ranking = np.zeros(len(ranking_ind))
        for r in range(len(ranking_ind)):
            ranking[r] = np.where(ranking_ind == r)[0][0]

        probabilities = np.exp(-10 * ranking / float(len(ranking))) / np.sum(np.exp(-10 * ranking / float(len(ranking))))
        probabilities = probabilities / np.sum(probabilities)

        # draw K samples according to the probabilities without replacement
        sampled_indices = np.random.choice(len(probabilities), K, replace=False, p=probabilities)
        sampled_new_points = val_centers[sampled_indices]


    # Perform Lloyd's relaxation
    if lloyd:
        norm_cur_centers = cur_centers / np.linalg.norm(cur_centers, axis=1, keepdims=True)
        norm_sampled_new_points = sampled_new_points / np.linalg.norm(sampled_new_points, axis=1, keepdims=True)
        new_points = lloyd_relaxation(norm_cur_centers, norm_sampled_new_points, niter=10, top=top, crop=crop)
    else:
        new_points = sampled_new_points / np.linalg.norm(sampled_new_points, axis=1, keepdims=True)
        
    point_to_add = new_points[-len(sampled_new_points):]

    return point_to_add