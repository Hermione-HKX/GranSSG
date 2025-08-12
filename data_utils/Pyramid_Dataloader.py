"""

"""
import copy
import glob
import os
import sys

sys.path.append('xxx')  # HACK add the lib folder
# import time
import copy
import random
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from tqdm import tqdm
from data_prepross_hkx import define
from torch_points_kernels import knn
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import data_utils.transform as t

sys.path.append(os.path.join(os.getcwd(), "../lib"))  # HACK add the lib folder


def vis(data, name):
    """ visualize the point cloud """
    if type(data) == type([]):
        data = np.asarray(data)
    np.savetxt(define.ROOT_PATH + name + '.txt', data)


def knn_speedup(points, query, k):
    if len(points.shape) == 2:
        points = points.reshape((1, -1, 3))
        query = query.reshape((1, -1, 3))
    idx, dist = knn(torch.tensor(points).double(), torch.tensor(query).double(), k)
    return idx.numpy(), dist.numpy()


class PyramidDataset(Dataset):
    def __init__(self, data_root, split="train", max_scene=10000, use_volume=True, random_max=False):
        ''' target: obtain all data path and split into train/val/test set '''
        self.split = split
        self.random_max = random_max
        train_or_val = 'train' if split == 'train' else 'val'
        self.max_scene = max_scene
        self.use_volume = use_volume
        self.min_obj_points = 128  # the min points contains in obj
        self.trans = t.Compose([t.Compose([t.RandomScale([0.9, 1.1]), t.RandomJitter(),
                                           t.ChromaticAutoContrast(),
                                           t.ChromaticTranslation(), t.ChromaticJitter(),
                                           t.HueSaturationTranslation()])])

        root = os.path.join(data_root, train_or_val)
        self.root = root
        self.npz_list = glob.glob(root + '/*/' + '/*.npz')
        if self.split == 'train':
            self.loop = 2
        else:
            self.loop = 1

    def __len__(self):
        return int(len(self.npz_list) * self.loop)
        # return 24

    def __getitem__(self, idx):
        """ build_in function: make this class can be indexed like list in python """
        idx = idx % len(self.npz_list)
        data_dict = np.load(self.npz_list[idx])

        sub_scene = data_dict["sub_scene"]
        objects_id = data_dict["objects_id"]
        edges = data_dict["edges"]
        objects_cat = data_dict["objects_cat"]
        predicate_cat = data_dict["predicate_cat"][:, 1:]

        sub_scene = self.point_trans(sub_scene, objects_id)
        objects_bbox, objects_hull_volume = self.bbox_msg(sub_scene, objects_id)
        if self.split != 'test_with_volume':
            return torch.tensor(sub_scene), torch.tensor(objects_bbox), torch.tensor(objects_id), torch.tensor(edges), \
                   torch.tensor(objects_cat), torch.tensor(predicate_cat)
        else:
            return torch.tensor(sub_scene), torch.tensor(objects_bbox), torch.tensor(objects_id), torch.tensor(edges), \
                torch.tensor(objects_cat), torch.tensor(predicate_cat), torch.tensor(objects_hull_volume)

    def point_trans(self, sub_scene, objects_id):

        # keep_ids = [np.where(sub_scene[:, -1] == x)[0][0] for x in objects_id]
        keep_idx = []
        for x in objects_id:
            if np.sum(sub_scene[:, -1] == x) < self.min_obj_points:
                instance = np.tile(sub_scene[sub_scene[:, -1] == x, :],
                                   (1 + int(self.min_obj_points / np.sum(sub_scene[:, -1] == x)), 1))
                sub_scene = np.vstack((sub_scene, instance))
            keep_idx.extend(np.where(sub_scene[:, -1] == x)[0][:self.min_obj_points])

        bool_idx = np.ones(sub_scene.shape[0], dtype=bool)
        bool_idx[keep_idx] = False
        bool_sub_scene = sub_scene[bool_idx]

        if self.random_max:
            max_scene = int(self.max_scene * random.uniform(0.8, 1.2))
        else:
            max_scene = self.max_scene

        if sub_scene.shape[0] > max_scene:  # too much
            choice_idx = np.random.choice(np.arange(0, bool_sub_scene.shape[0]), max_scene - len(keep_idx),
                                          replace=False)
            keep_points = sub_scene[keep_idx]
            sub_scene = sub_scene[choice_idx]
            sub_scene = np.vstack((sub_scene, keep_points))

        # ==== transformer ====
        # np.savetxt('org.txt', sub_scene)
        # coord, feat = self.trans(sub_scene[:, :3], sub_scene[:, 3:])
        if self.split == 'train':
            coord, feat = self.trans(copy.deepcopy(sub_scene[:, :3]), copy.deepcopy(sub_scene[:, 3:]))
            sub_scene = np.hstack([coord, feat])
        # np.savetxt('trans.txt', sub_scene)
        # =====================

        min_xyz = np.min(sub_scene[:, :3], axis=0)
        sub_scene[:, :3] = sub_scene[:, :3] - min_xyz
        sub_scene[:, 3:6] = sub_scene[:, 3:6] / 255.

        return sub_scene

    def bbox_msg(self, sub_scene, objects_id):
        """ build the bbox message """
        objects_bbox = []
        objects_volume = []
        for obj_mask in objects_id:
            obj = sub_scene[sub_scene[:, -1] == obj_mask]
            try:
                obj_o3d = o3d.geometry.PointCloud()
                obj_o3d.points = o3d.utility.Vector3dVector(obj[:, :3])
                obb_box = obj_o3d.get_oriented_bounding_box()

                convex_hull, _ = obj_o3d.compute_convex_hull()
                convex_surface = convex_hull.get_surface_area()

                hull = ConvexHull(obj[:, :3])
                volume = hull.volume

                obb_msg = np.hstack((obb_box.center, obb_box.extent, np.log(np.prod(obb_box.extent)),
                                     np.log(volume), np.log(convex_surface)))
            except:
                # don't have enough points build the bbox: xyz-->center, extent-->0.002, vb,vs--> 1e-5
                center = np.mean(obj[:, :3], axis=0)
                obb_msg = np.hstack((center, np.ones((3,)) * 0.002, np.log(1e-5), np.log(1e-5), np.log(1e-5)))
                volume = 1e-5
            objects_bbox.append(obb_msg)
            objects_volume.append(volume)
        objects_bbox = np.vstack(objects_bbox)
        objects_volume = np.vstack(objects_volume)

        if self.use_volume:
            return objects_bbox, objects_volume
        else:
            return objects_bbox[:, :7], objects_volume

def offset_cal(pc):
    offset, count = [], 0
    for item in pc:
        count += item.shape[0]
        offset.append(count)
    return torch.IntTensor(offset)


def offset_edges(edges):
    offset, count = [], 0
    for edge in edges:
        # assert torch.max(edge) * (torch.max(edge) + 1) == edge.shape[0]
        offset.extend(edge + count)
        count += (torch.max(edge) + 1)
    return torch.vstack(offset)


def collate_fn(batch):
    if len(batch[0]) == 6:
        scene_pc, objects_bbox, objects_id, edges, objects_cat, predicate_cat = list(zip(*batch))
        offset_pair = offset_cal(scene_pc)
        edges = offset_edges(edges)
        return [torch.cat(scene_pc), torch.cat(objects_bbox), objects_id, edges,
                torch.cat(objects_cat), torch.cat(predicate_cat), offset_pair]
    else:
        scene_pc, objects_bbox, objects_id, edges, objects_cat, predicate_cat, volume = list(zip(*batch))
        offset_pair = offset_cal(scene_pc)
        edges = offset_edges(edges)
        return [torch.cat(scene_pc), torch.cat(objects_bbox), objects_id, edges,
                torch.cat(objects_cat), torch.cat(predicate_cat), offset_pair, torch.vstack(volume)]

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_path = 'xxx'
    train_dataset = PyramidDataset(data_root=data_path, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn,
                                  num_workers=8, drop_last=True)
    a = 0
    for data in tqdm(train_dataloader):
        a += 1
    print('end')
