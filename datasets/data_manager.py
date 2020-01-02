from __future__ import print_function, absolute_import

import glob
import re
from os import path as osp
import os

"""Dataset classes"""


class SKU:
    def __init__(self, train_indices, gallery_indices, query_indices, class_list_path=None):
        # self.image_dir = image_dir
        self.train, self.num_train_pids, self.num_train_imgs = self._read_indices(train_indices, init_camid=0)
        self.gallery, self.num_gallery_pids, self.num_gallery_imgs = self._read_indices(gallery_indices, class_list_path=class_list_path, init_camid=0)
        self.query, self.num_query_pids, self.num_query_imgs = self._read_indices(query_indices, class_list_path=class_list_path, init_camid=100000000)

    def _read_indices(self, path, class_list_path=None, init_camid=0):
        with open(path, "r") as f:
            data = [l.strip().split() for l in f]
        all_labels = sorted({d[1] for d in data})
        if class_list_path is None:
            label2id = {label: _id for _id, label in enumerate(all_labels)}
        else:
            with open(class_list_path, "r") as f:
                label2id = {c.strip(): i for i, c in enumerate(f)}

        # pretend all the images are taken from different cameras to make the code work.
        return [(d[0], label2id[d[1]], init_camid + i) for i, d in enumerate(data)], len(all_labels), len(data)
        # return [(f"{self.image_dir}/{d[0]}", label2id[d[1]], None) for d in data], len(all_labels), len(data)


class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, dataset_dir, mode, root='data'):
        self.dataset_dir = dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        train_relabel = (mode == 'retrieval')
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=train_relabel)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_names = os.listdir(dir_path)
        img_paths = [os.path.join(dir_path, img_name) for img_name in img_names \
            if img_name.endswith('jpg') or img_name.endswith('png')]
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

def init_dataset(name, mode):
    return Market1501(name, mode)
