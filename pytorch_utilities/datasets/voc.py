import os
import sys
import tarfile
import collections
from torchvision.datasets import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg


DATASET_YEAR_DICT = {
    '2012': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            'filename': 'VOCtrainval_11-May-2012.tar',
            'md5': '6cd6e144f989b92b3379bac3b3de84fd',
            'base_dir': 'VOCdevkit/VOC2012'
        },
        'test': {
            'url': 'http://pjreddie.com/media/files/VOC2012test.tar',
            'filename': 'VOC2012test.tar',
            'md5': '',
            'base_dir': 'VOCdevkit/VOC2012'
        }
    },
    '2007': {
        'trainval': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'filename': 'VOCtrainval_06-Nov-2007.tar',
            'md5': 'c52e279531787c972589f7e41ab4ae64',
            'base_dir': 'VOCdevkit/VOC2007'
        },
        'test': {
            'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'filename': 'VOCtest_06-Nov-2007.tar',
            'md5': 'b6e924de25625d8de591ea690078ad9f',
            'base_dir': 'VOCdevkit/VOC2007'
        }
    }
}


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 and 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval``, ``val``, and ``test``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2007',
                 image_set='trainval',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        tgt = 'test' if image_set == 'test' else 'trainval'
        self.year = year
        self.url = DATASET_YEAR_DICT[year][tgt]['url']
        self.filename = DATASET_YEAR_DICT[year][tgt]['filename']
        self.md5 = DATASET_YEAR_DICT[year][tgt]['md5']
        self.image_set = image_set

        base_dir = DATASET_YEAR_DICT[year][tgt]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


def fetch_region_proposals(source_dir='./data/proposals/'):
    region_proposals = {}
    for file in ['voc_2007_train.mat', 'voc_2007_val.mat', 'voc_2007_test.mat']:
        mat = io.loadmat(os.path.join(source_dir, file))
        boxes = mat['boxes'][0] if isinstance(mat['boxes'], np.ndarray) else mat['boxes']

        if file == 'voc_2007_test.mat':
            images = [fid[0] for fid in mat['images']]
        else:
            images = mat['images'][0]

        for file_id, rois in zip(images, boxes):
            rois = (rois - 1).astype(float)[:,[1,0,3,2]]
            region_proposals[file_id[0]] = rois
    return region_proposals


def extract_image_labels(target_dict, class_names):
    objects = target_dict['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    label = torch.zeros((len(class_names)-1,), dtype=torch.float)
    for obj in objects:
        label[class_names.index(obj['name'][0])] = 1.
    return label.unsqueeze(0)


def extract_box_labels(target_dict, class_names):
    objects = target_dict['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    rois, classes = [], []
    for obj in objects:
        roi = obj['bndbox']
        roi = torch.tensor([float(roi['xmin'][0]),
                            float(roi['ymin'][0]),
                            float(roi['xmax'][0]),
                            float(roi['ymax'][0])])
        cls = class_names.index(obj['name'][0])

        rois.append(roi)
        classes.append(cls)

    return torch.stack(rois), torch.tensor(classes, dtype=torch.long)


def extract_file_id(target_dict):
    return target_dict['annotation']['filename'][0][:-4]
