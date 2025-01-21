import json
import os
import argparse
import pickle

import numpy as np

np.random.seed(42)

from internal.colmap_utils import get_3d_correspondence_matrix
from internal.dataset_utils import *
from internal.fvs import farthest_view_sampling, farthest_view_sampling_colmap
from internal.vmf import vMF_sampling
from internal.zipf import zipf_sampling


# [Optional] evaluation with instant-ngp uncomment the following lines
# import sys
# sys.path.append('./instant-ngp/scripts')

from train_and_eval_ingp import run_ngp

""" NeRF Director main script """

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for NeRF Director")

    # Experiment information
    parser.add_argument("--model", default="chair", type=str, 
                        help="scene name")
    # parser.add_argument("--exp", default="random_sampling", type=str, 
    #                     help="experiment name")
    parser.add_argument("--rep", default=0, type=int,
                        help="current repetition")
    parser.add_argument("--sampling", default="fvs", 
                        help="sampling method, including rs, fvs, zipf, vmf, greedy")
    parser.add_argument("--checkpoint_dir", default="", type=str, 
                        help="directory to store all training checkpoints and test results")
    parser.add_argument("--eval_all", action="store_true",
                        help="indicate if running all the training views of a split")
    
    # Dataset
    parser.add_argument("--all_train_transform", default="", type=str, 
                        help="path to transforms_train.json of the entire training images")
    parser.add_argument("--test_transform", default="", type=str, 
                        help="path to transforms_test.json")
    parser.add_argument("--real_world_data", action="store_true",
                        help="indicate if current dataset is real-world one")
    
    # Farthest view sampling related
    parser.add_argument("--dist_type", default="euc", type=str,
                        help="distance type for farthest view sampling, inc. euc, gcd")
    parser.add_argument("--enable_photo_dist", action="store_true",
                        help="specify if using photometric distance for fvs")
    parser.add_argument("--colmap_log", default='', type=str,
                        help="colmap images.txt file path")
    parser.add_argument("--use_val", action="store_true",
                        help="use the information gain based on test set otherwise unselected training candidates")
    
    # Information gain-based sampling related
    parser.add_argument("--ig_type", default='psnr', type=str, 
                        help="information type used for information gain-based sampling")
    parser.add_argument("--lloyd", action="store_true",
                        help="specify if applying lloyd relaxation")

    # Von Mises-Fisher distribution related
    parser.add_argument("--vmf_sigma", default=0.25, type=float,
                        help="sigma for PSNR balance in vMF")
    parser.add_argument("--vmf_kappa", default=100, type=float,
                        help="kappa concentration for von Mises-Fisher distribution")

    return parser.parse_args()

def print_log(str):
    print('************************************************************************')
    print(str)
    print('************************************************************************\n')

def print_args(arg):
    str = 'EXPERIEMENT ARGUMENTS\n'
    
    for a in vars(arg).keys():
        str += '* {}: {}\n'.format(a, vars(arg)[a])
    
    print_log(str)



def run_train_and_eval(cur_train_transform, val_transform, ckpt_dir, view_num, args, eval_val=True):
    ''' Train and eval current configuration

    Args:
        cur_train_transform: string, path to current training transform
        val_transform: string, path to transform of validation points
        ckpt_dir: string, path to the checkpoint folder storing training, val, and testing results
        view_num: int, current number of training views
        args: dict, arguments of experiments
        eval_val: bool, indicating if evaluating the validation set
    
    Returns:
        psnrs: list, all frames' psnr
        ssims: list, all frames' ssim
        full_masked_mses: list, all frames' mse on the mask of the union of both g.t. and recon denstiy
        ref_masked_mses: list, all frames' mse on the mask of g.t. density map
    '''
    # checkpoint setting
    ckpt = os.path.join(ckpt_dir, 'checkpoint_{}.ingp'.format(str(view_num).zfill(3)))

    # training
    args_ngp = pickle.load(open('./configs/ingp/args_ngp.pkl', 'rb'))
    if args.real_world_data:
        args_ngp.files = [cur_train_transform, './configs/ingp/base_tnt.json']
        args_ngp.nerf_compatibility = False
        args_ngp.n_steps = 50000
    else:
        args_ngp.files = [cur_train_transform]
    args_ngp.save_snapshot = ckpt
    args_ngp.train_dir = ckpt_dir
    run_ngp(args_ngp)
    
    # evaluation on the test set
    test_screenshot_dir = os.path.join(ckpt_dir, 'test')
    if not os.path.exists(test_screenshot_dir):
        os.makedirs(test_screenshot_dir)
    args_ngp = pickle.load(open('./configs/ingp/args_ngp.pkl', 'rb'))
    args_ngp.files = []
    args_ngp.load_snapshot = ckpt
    args_ngp.screenshot_dir = test_screenshot_dir
    args_ngp.write_image = True
    args_ngp.test_transforms = args.test_transform
    if args.real_world_data:
        args_ngp.nerf_compatibility = False
    run_ngp(args_ngp)

    # evaluation on the validation set
    psnrs, ssims, full_masked_mses, ref_masked_mses = [], [], [], []
    if eval_val:
        val_screenshot_dir = os.path.join(ckpt_dir, 'val')
        if not os.path.exists(val_screenshot_dir):
            os.makedirs(val_screenshot_dir)
        args_ngp = pickle.load(open('./configs/ingp/args_ngp.pkl', 'rb'))
        args_ngp.files = []
        args_ngp.load_snapshot = ckpt
        args_ngp.screenshot_dir = val_screenshot_dir
        args_ngp.test_transforms = val_transform
        if args.real_world_data:
            args_ngp.nerf_compatibility = False
        _, _, psnrs, ssims, full_masked_mses, ref_masked_mses, _ = run_ngp(args_ngp)

    return psnrs, ssims, full_masked_mses, ref_masked_mses

def update_available_list(current_centers, all_centers, available_list):
    new_view_ids = []
    
    # normalize all the center
    norm_current_centers = current_centers / np.linalg.norm(current_centers, axis=1, keepdims=True)
    norm_all_centers = all_centers / np.linalg.norm(all_centers, axis=1, keepdims=True)
    
    for selected_center in norm_current_centers:
        dist = np.linalg.norm(norm_all_centers - selected_center, axis=1)
        dist[~available_list] = np.inf
        id_selected = np.argmin(dist)
        new_view_ids.append(id_selected)
        available_list[id_selected] = False

    return new_view_ids


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print_log('{} not exists! Create a new directory!'.format(args.checkpoint_dir))

        os.makedirs(args.checkpoint_dir)

    if args.ig_type not in ['psnr', 'ssim']:
        args.ig_type = 'psnr'
    
    if args.sampling not in ['zipf', 'vmf', 'fvs', 'rs', 'greedy']:
        args.sampling = 'fvs'
    
    if args.sampling == 'greedy':
        args.lloyd = False
    
    if args.enable_photo_dist and not os.path.exists(args.colmap_log):
        print_log('Required COLMAP log does not exist! Turn off photometric distance for FVS.')

        args.enable_photo_dist = False
    
    print_args(args)


    top = False if args.model == 'ficus' or args.model == 'materials' or args.real_world_data else True

    # fetch all available training view candidates
    all_available_candidates = get_camera_centers(args.all_train_transform)
    is_available = np.ones(len(all_available_candidates), dtype=bool)
    
    
    # fetch validation view centers
    val_transform = ""
    if args.use_val:
        eval_centers = get_camera_centers(args.test_transform)
        val_transform = args.test_transform

    view_num_configs = [
        5,      10,     15,     20,     25,     30,     40,     50,     60,
        70,     80,     90,     100,    110,    120,    130,    140,    150
    ]

    psnrs, ssims, full_masked_mses, ref_masked_mses, guide_info  = [], [], [], [], []

    # generate the 3d correspondence matrix
    if args.sampling == 'fvs' and args.enable_photo_dist:
        print('Generate D matrix for real-world data...')
        
        # get image_names, cur_split_all_inds, and D        
        with open(args.all_train_transform, 'r') as base_json:
            base_data = json.load(base_json)
            image_names = [ os.path.basename(x['file_path']) for x in base_data['frames'] ]
    
        # construct the 3d correspondence matrix D for real-world data
        D = get_3d_correspondence_matrix(args.colmap_log, image_names)
    

    
    # run all train candidates
    if args.eval_all:
        print_log("Evaluating all training views...")
        psnrs, ssims, full_masked_mses, ref_masked_mses = run_train_and_eval(
            args.all_train_transform, 
            args.test_transform, 
            os.path.dirname(args.all_train_transform),
            len(all_available_candidates), 
            args=args, eval_val=False
        )
    
    
    last_view_num = 0
    for i, cur_view_num in enumerate(view_num_configs):
        print_log('Running sampling={} view_number={}'.format(
            args.sampling, cur_view_num)
        )

        cur_checkpoint_dir = os.path.join(args.checkpoint_dir, '{}'.format(cur_view_num))
        if not os.path.exists(cur_checkpoint_dir):
            print_log('{} does not exisits! Create a new checkpoint folder!'.format(
                cur_checkpoint_dir
            ))
            os.makedirs(cur_checkpoint_dir)
        
        # initial fps setting
        cur_train_transform = os.path.join(cur_checkpoint_dir, 'transforms_train.json')

        # modify the relative path if initial fps transform is provided
        # otherwise, generate a new one
        if last_view_num == 0:
            # create initial points through random sampling
            np.random.seed(args.rep)
            new_views = np.random.randint(
                0, high=len(all_available_candidates) - 1, size=cur_view_num
            )

            generate_new_transform(
                "", cur_train_transform, 
                args.all_train_transform, new_views
            )

            # fetch current training view centers
            current_centers = get_camera_centers(cur_train_transform)
            update_available_list(
                current_centers, all_available_candidates, is_available
            )

        else:

            if args.sampling == "zipf":
                crop = True if 'tnt' in args.all_train_transform else False
                new_views = zipf_sampling(
                    cur_view_num - last_view_num, guide_info, 
                    current_centers, eval_centers, args.ig_type,
                    top=top, lloyd=args.lloyd, crop=crop
                )
                new_indices = update_available_list(
                    new_views, all_available_candidates, is_available
                )
                generate_new_transform(
                    last_train_transform, cur_train_transform, 
                    args.all_train_transform, new_indices
                )
                current_centers = all_available_candidates[~is_available]
                
            elif args.sampling == "rs":
                np.random.seed(args.rep * cur_view_num)
                new_indices = np.arange(0, len(all_available_candidates))
                new_indices = new_indices[is_available]
                np.random.shuffle(new_indices)
                new_indices = new_indices[:cur_view_num - last_view_num]
                is_available[new_indices] = False
                generate_new_transform(
                    last_train_transform, cur_train_transform, 
                    args.all_train_transform, new_indices
                )
                
            elif args.sampling == "vmf":
                crop = True if 'tnt' in args.all_train_transform else False
                new_views = vMF_sampling(
                    cur_view_num - last_view_num, guide_info, 
                    current_centers, eval_centers, 
                    sigma=args.vmf_sigma, kappa=args.vmf_kappa, 
                    top=top, lloyd=args.lloyd, crop=crop
                )
                new_indices = update_available_list(
                    new_views, all_available_candidates, is_available
                )
                generate_new_transform(
                    last_train_transform, cur_train_transform, 
                    args.all_train_transform, new_indices
                )
                current_centers = all_available_candidates[~is_available]

            elif args.sampling == "fvs":
                if args.enable_photo_dist:
                    new_views = farthest_view_sampling_colmap(
                        cur_view_num - last_view_num, all_available_candidates,
                        args.rep, D, dist_type=args.dist_type, 
                        selected_status=~is_available
                    )
                else:
                    new_views = farthest_view_sampling(
                        cur_view_num - last_view_num, all_available_candidates,
                        args.rep, dist_type=args.dist_type, 
                        selected_status=~is_available
                    )
                new_indices = np.array(new_views[last_view_num:], dtype=np.int32)
                available_new_indices = np.unique(new_indices[is_available[new_indices]])
                new_indices = available_new_indices[:cur_view_num - last_view_num]
                is_available[new_indices] = False
                generate_new_transform(
                    last_train_transform, cur_train_transform, 
                    args.all_train_transform, new_indices
                )

            elif args.sampling == "greedy":
                new_views = zipf_sampling(
                    cur_view_num - last_view_num, guide_info, 
                    current_centers, eval_centers, 
                    args.ig_type, top=top, crop=False, 
                    lloyd=False, strict=True
                )
                new_indices = update_available_list(
                    new_views, all_available_candidates, is_available
                )
                generate_new_transform(
                    last_train_transform, cur_train_transform, 
                    args.all_train_transform, new_indices
                )
                current_centers = all_available_candidates[~is_available]
            
        # select validation transform
        if args.sampling in ['zipf', 'vmf', 'greedy'] and not args.use_val:
            val_transform = os.path.join(cur_checkpoint_dir, "transforms_val.json")
            eval_inds = np.arange(len(all_available_candidates))
            eval_inds = eval_inds[is_available]
            generate_new_transform(
                "", val_transform, 
                args.all_train_transform, eval_inds
            )
            
            # get sorted eval centers
            with open(val_transform, 'r') as f:
                val_data = json.load(f)
            if args.real_world_data:
                val_file_path = np.array([ os.path.basename(f['file_path']) for f in val_data['frames'] ])
            else:
                val_file_path = np.array([ int(f['file_path'].split('_')[-1]) for f in val_data['frames'] ])
            eval_centers = all_available_candidates[is_available]
            eval_centers = eval_centers[np.argsort(val_file_path)]

        print_log("Current train transform: {}".format(cur_train_transform))


        eval_val = True if args.sampling in ['zipf', 'vmf', 'greedy'] else False
        psnrs, ssims, full_masked_mses, ref_masked_mses = run_train_and_eval(
            cur_train_transform, val_transform, cur_checkpoint_dir,
            cur_view_num, args, eval_val=eval_val
        )

        if args.ig_type == "psnr":
            guide_info = psnrs
        elif args.ig_type == "ssim":
            guide_info = ssims

            
        last_train_transform = cur_train_transform
        last_view_num = cur_view_num

