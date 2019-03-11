from __future__ import absolute_import
from __future__ import print_function

import os.path as osp


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def __init__(self, root):
        self.root = osp.expanduser(root)

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print('Image Dataset statistics:')
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, num_train_imgs, num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, num_query_imgs, num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print('  ----------------------------------------')
