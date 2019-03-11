from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset
from .datasets import init_imgreid_dataset
from .transforms import build_transforms
from .samplers import build_train_sampler
from .utils.mean_and_std import  get_mean_and_std, calculate_mean_and_std


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root='datasets',
                 height=128,
                 width=256,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 random_erase=False,  # use random erasing for data augmentation
                 color_jitter=False,  # randomly change the brightness, contrast and saturation
                 color_aug=False,  # randomly alter the intensities of RGB channels
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances

        transform_train, transform_test = build_transforms(
            self.height, self.width, random_erase=self.random_erase, color_jitter=self.color_jitter,
            color_aug=self.color_aug
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Vehicle-ReID data manager
    """
    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)

        print('=> Initializing TRAIN (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name)

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train, self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train), sampler=self.train_sampler,
            batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True
        )
        mean, std = calculate_mean_and_std(self.trainloader, len(train))
        print('mean and std:', mean, std)

        print('=> Initializing TEST (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name)

            self.testloader_dict[name]['query'] = DataLoader(
                ImageDataset(dataset.query, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                ImageDataset(dataset.gallery, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.source_names))
        print('  # train datasets : {}'.format(len(self.source_names)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(train)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test names       : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')
