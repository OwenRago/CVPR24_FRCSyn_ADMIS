import os
import logging
import torch.distributed as dist
from torch.utils.data import Dataset
from .parser import IndexParser, ImgSampleParser, TFRecordSampleParser, Synthetic_IndexParser, AdaFaceTFRecordSampleParser

class SingleDataset(Dataset):
    """ SingleDataset
    """

    def __init__(self, data_root, index_root, name, transform, **kwargs) -> None:

        self.image_mode = os.path.isdir(self.data_root)

        """ Create a ``SingleDataset`` object

            Args:
            data_root: image or tfrecord data root path
            index_root: index file root path
            name: dataset name
            transform: transform for data augmentation
        """

        super().__init__()
        self.data_root = data_root
        self.index_root = index_root
        self.name = name
        self.index_parser = IndexParser()
        if 'TFR' not in name and 'TFR' not in kwargs:
            self.sample_parser = ImgSampleParser(transform)
        else:
            self.sample_parser = TFRecordSampleParser(transform)

        self.inputs = []
        self.is_shard = False
        self.class_num = 0
        self.sample_num = 0

    def make_dataset(self):
        self._build_inputs()

    def _build_inputs(self):
        """ Read index file and saved in ``self.inputs``
        """

        index_path = os.path.join(self.index_root, self.name + '.txt')
        with open(index_path, 'r') as f:
            for line in f:
                sample = self.index_parser(line)
                self.inputs.append(sample)
        self.class_num = self.index_parser.class_num + 1
        self.sample_num = len(self.inputs)
        logging.info("Dataset %s, class_num %d, sample_num %d" % (
            self.name, self.class_num, self.sample_num))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        """ Parse image and label data from index
        """

        sample = list(self.inputs[index])
        sample[0] = os.path.join(self.data_root, sample[0])  # data_root join
        image, label = self.sample_parser(*sample)
        return image, label


class MultiDataset(Dataset):
    """ MultiDataset, which contains multiple dataset and should be used
        together with ``MultiDistributedSampler``
    """

    def __init__(self, data_root, index_root, names, transform, num_duplices=1, **kwargs) -> None:
        """ Create a ``MultiDataset`` object

            Args:
            data_root: image or tfrecord data root path
            index_root: index file root path
            name: dataset names for multiple dataset
            transform: transform for data augmentation
        """

        super().__init__()
        self.data_root = data_root
        self.index_root = index_root
        self.names = names
        self.index_parser = IndexParser()
        self.num_duplices = num_duplices
        if 'TFR' not in names[0] and 'TFR' not in kwargs:
            print('Image Index Parser')
            self.sample_parser = ImgSampleParser(transform)
        else:
            print('TFR Index Parser')
            self.sample_parser = TFRecordSampleParser(transform)

        self.inputs = dict()
        self.is_shard = False
        self.class_nums = dict()
        self.sample_nums = dict()

    @property
    def dataset_num(self):
        return len(self.names)

    @property
    def class_num(self):
        class_nums = []
        for name in self.names:
            class_nums.append(self.class_nums[name])
        return class_nums

    def make_dataset(self, shard=False):
        """ If shard is True, the inputs are shared into all workers for memory efficiency.
            If shard is False, each worker contain the total inputs.
        """

        if shard:
            self.is_shard = True
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self.total_sample_nums = dict()
            self._build_inputs(world_size, rank)
        else:
            self._build_inputs()

    def _build_inputs(self, world_size=None, rank=None):
        """ Read index file and saved in ``self.inputs``
            If ``self.is_shard`` is True, ``total_sample_nums`` > ``sample_nums``
        """

        for i, name in enumerate(self.names):
            # index_file = os.path.join(self.index_root, name + ".txt")
            index_file = os.path.join(self.data_root, "..", "tfrecords", name + ".index")
            self.index_parser.reset()
            self.inputs[name] = []
            with open(index_file, 'r') as f:
                for line_i, line in enumerate(f):
                    sample = self.index_parser(line)
                    if self.is_shard is False:
                        for i in range(self.num_duplices):
                            self.inputs[name].append(sample)
                    else:
                        if line_i % world_size == rank:
                            self.inputs[name].append(sample)
                        else:
                            pass
            self.class_nums[name] = self.index_parser.class_num + 1
            self.sample_nums[name] = len(self.inputs[name])
            if self.is_shard:
                self.total_sample_nums[name] = self.index_parser.sample_num
                logging.info("Dataset %s, class_num %d, total_sample_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.total_sample_nums[name], self.sample_nums[name]))
            else:
                logging.info("Dataset %s, class_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.sample_nums[name]))

    def __getitem__(self, index):
        """ Parse image and label data from index,
            the index size should be equal with 2
        """
        if len(index) != 2:
            raise RuntimeError("MultiDataset index size wrong")
        else:
            name, index = index
            sample = list(self.inputs[name][index])
            sample[0] = os.path.join(self.data_root, sample[0])  # data_root join
            image, label = self.sample_parser(*sample)
        return image, label

# For CASIA baseline
class CASIADataset(Dataset):
    def __init__(self, data_root, name, transform, num_duplices=1, AdaFace_augment_prob = 0.2, clean_txt = None, **kwargs) -> None:
            """ Create a ``SingleDataset`` object

                Args:
                data_root: image or tfrecord data root path
                stylebank_suffix: {index}_stylebank_BUPT_10k.jpg
                name: dataset name
                transform: transform for data augmentation
            """
            super().__init__()
            self.data_root = data_root
            self.names = name
            self.num_duplices = num_duplices
            self.sample_parser = ImgSampleParser(transform, crop_augmentation_prob = AdaFace_augment_prob)
            self.inputs = []
            self.is_shard = False
            self.class_nums = dict()
            self.sample_nums = dict()
            self.clean_txt = clean_txt
            self._build_inputs()

    @property
    def dataset_num(self):
        return len(self.names)

    @property
    def class_num(self):
        class_nums = []
        for name in self.names:
            class_nums.append(self.class_nums[name])
        return class_nums

    def make_dataset(self):
        self._build_inputs()

    def _build_inputs(self):
        for i, name in enumerate(self.names):
            self.inputs[name] = []

            # ✅ Correct the path to the .index file
            index_file = os.path.join(self.data_root, "..", "tfrecords", name + ".index")
            index_file = os.path.normpath(index_file)  # clean up the path

            if not os.path.isfile(index_file):
                raise FileNotFoundError(f"[ERROR] Index file not found: {index_file}")

            d = self.build_dict(index_file)

            for key in d:
                label = int(key.split("/")[-2])
                tfr_name, file_index, file_offset = d[key].split("\t")
                sample = self.index_parser(label, tfr_name, file_index, file_offset)
                self.inputs[name].append(sample)

            self.class_nums[name] = self.index_parser.class_num + 1
            self.sample_nums[name] = len(self.inputs[name])
            logging.info("Dataset %s, class_num %d, sample_num %d" % (
                name, self.class_nums[name], self.sample_nums[name]))


    def __len__(self):
        return len(self.sample_num)

    def __getitem__(self, index):
        """ Parse image and label data from index
        """
        label = self.inputs[index[1]][0]
        image_id_suffix = self.inputs[index[1]][1]

        image_path = os.path.join(self.data_root, image_id_suffix)
        sample = [image_path, label]
        image, label = self.sample_parser(*sample)
        return image, label



# For TFrecord-Synthetic data
class TF_SyntheticDataset(Dataset):
    def __init__(self, data_root, name, transform, num_duplices=1, AdaFace_augment_prob = 0.2, **kwargs) -> None:
        super().__init__()
        self.data_root = data_root
        self.names = name
        self.num_duplices = num_duplices
        self.sample_parser = ImgSampleParser(transform, crop_augmentation_prob = AdaFace_augment_prob)
        self.inputs = dict()
        self.is_shard = False
        self.class_nums = dict()
        self.sample_nums = dict()
        self._build_inputs()
        self.all_samples = []
        for name in self.names:
            self.all_samples.extend(self.inputs[name])


    @property
    def dataset_num(self):
        return len(self.names)

    @property
    def class_num(self):
        return [self.class_nums[name] for name in self.names]

    def make_dataset(self):
        self._build_inputs()

    def _build_inputs(self):
        import glob
        inputs = []
        label_map = {}
        label_counter = 0

        for name in self.names:
            dataset_path = os.path.join(self.data_root, name)
            print(f"[INFO] Loading images from: {dataset_path}")

            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"Expected image folder at: {dataset_path}")

            image_paths = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
            if not image_paths:
                raise ValueError(f"No images found in {dataset_path}")

            for img_path in image_paths:
                # ❗ Get the immediate folder name under 'all', e.g., class0, class1, etc.
                class_name = os.path.basename(os.path.dirname(img_path))

                if class_name not in label_map:
                    label_map[class_name] = label_counter
                    label_counter += 1

                inputs.append([img_path, label_map[class_name]])

        self.inputs = {"all": inputs}
        self.class_nums = {"all": len(label_map)}
        self.sample_nums = {"all": len(inputs)}

        print(f"[INFO] Loaded {len(inputs)} images with {len(label_map)} classes total.")
        print(f"[DEBUG] Label map: {label_map}")  # ← optional for checking

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        print(f"Index type: {type(index)}, Index value: {index}")
        
        # Handle tuple index
        if isinstance(index, tuple):
            key, index = index  # Unpack the tuple
            if key != "all":
                raise ValueError(f"Unexpected key in index: {key}. Expected 'all'.")
        
        # Use the integer index to access self.all_samples
        img_path, label = self.all_samples[index]
        print(f"Image path: {img_path}, Label: {label}")  # Debug print

        sample = [img_path, label]
        image, label = self.sample_parser(*sample)
        print(f"Image shape: {image.shape}, Label after parsing: {label}")  # Debug print
        return image, label


