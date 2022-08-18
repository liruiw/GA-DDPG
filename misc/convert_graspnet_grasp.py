
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import time
import argparse
import sys
sys.path.insert(0, '../pytorch_6dof-graspnet')
from grasp_data_reader import *
import IPython

object_root_folder = '../data/objects/'
grasp_root_folder  = '../data/grasps/simulated/'
def mkdir_if_missing(dst_dir):
  if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)


def convert_file(file, mesh, grasp):
    names = file.split('/')
    category = names[-2].split('_')[0]


    object_id = names[-1].split('.')[-2].split('_')[-1]
    obj_folder = os.path.join(object_root_folder, category + '_' + object_id)
    file_name = 'model_normalized.obj'
    grasp_file_name = category + '_' + object_id + '.npy'
    # print('category: {} id: {} folder: {}'.format(category, object_id, obj_folder))
    # print('grasp shape:', grasp.shape)
    mkdir_if_missing(obj_folder)

    print('convert mesh to:', os.path.join(obj_folder, file_name))
    mesh.export(os.path.join(obj_folder, file_name), file_type='obj')

    # move forward a bit
    grasp[:, :3, 3] += grasp[:, :3, :3].dot(np.array([0, 0, 0.02]) )
    np.save(os.path.join(grasp_root_folder, grasp_file_name), {'transforms': grasp })



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument(
            '--root-folder',
            help='Root dir for data',
            type=str,
            default='/home/liruiw/Projects/pytorch_6dof-graspnet/unified_grasp_data')
    parser.add_argument(
            '--vae-mode',
            help='True for vae mode',
            action='store_true',
            default=False)
    parser.add_argument(
            '--grasps-ratio',
            help='ratio of grasps to be used from each cluster. At least one grasp is chosen from each cluster.',
            type=float,
            default=1.0
    )
    parser.add_argument(
        '--balanced_data',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--category_name',
        help='name of the category that needs to be converted.',
        type=str,
        default='mug'
    )
    parser.add_argument('--allowed_category', default='', type=str)

    args = parser.parse_args()
    args.root_folder = os.path.abspath(args.root_folder)
    print('Root folder', args.root_folder)

    import glob
    from visualization_utils import draw_scene
    import mayavi.mlab as mlab

    pcreader = PointCloudReader(
        root_folder=args.root_folder,
        batch_size=100,
        num_grasp_clusters=4, # 32
        npoints=1024,
        ratio_hardnegative=0,
        ratio_positive = 1,
        ratio_of_grasps_used=args.grasps_ratio,
        balanced_data=args.balanced_data
    )

    grasp_paths = glob.glob(os.path.join(args.root_folder, 'grasps') + '/*.json') # *{}.format(args.category_name))
    # grasp_paths = [g for g in grasp_paths if args.category_name in g]

    if args.allowed_category != '':
        grasp_paths = [g for g in grasp_paths if g.find(args.allowed_category)>=0]

    for grasp_path in grasp_paths:
        try:
            output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_files, output_cad_scales, obj = \
                 pcreader.get_evaluator_data(grasp_path, verify_grasps=False, transform=False)
        except:
            continue
        print(output_grasps.shape)

        for pc, pose in zip(output_pcs, output_pc_poses):
            assert(np.all(pc == output_pcs[0]))
            assert(np.all(pose == output_pc_poses[0]))


        pc = output_pcs[0]
        pose = output_pc_poses[0]
        cad_file = output_cad_files[0]
        cad_scale = output_cad_scales[0]

        # obj = sample.Object(cad_file)
        # obj.rescale(cad_scale)
        # obj = obj.mesh
        # obj.vertices -= np.expand_dims(np.mean(obj.vertices, 0), 0)

        convert_file(cad_file, obj, output_grasps)
        print('mean_pc', np.mean(pc, 0))
        print('pose', pose)
        print(obj.vertices.max(0))
        # print(output_labels)
        # draw_scene(
        #     pc,
        #     grasps=output_grasps,
        #     grasp_scores=None if args.vae_mode else output_labels,
        # )
        # mlab.figure()
        # draw_scene(
        #      pc.dot(pose.T),
        #     grasps= output_grasps, # [pose.dot(g) for g in output_grasps],
        #     mesh=obj,
        #     grasp_scores=None if args.vae_mode else output_labels,
        # )
        # mlab.show()
