import pytorch_lightning as pl
import typing as th
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from .utils import initialize_transforms
from torchde.utils import get_value

class CausalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # dataset
        dataset: th.Union[str, th.List["ocd.data.CausalDataset"]],
        dataset_args: th.Optional[dict] = None,
        # validation
        val_size: th.Optional[th.Union[int, float]] = None,
        test_dataset: th.Optional[th.Union[str, th.List["ocd.data.CausalDataset"]]] = None,
        test_dataset_args: th.Optional[dict] = None,
        # transforms
        transforms: th.Optional[th.Union[dict, th.Any]] = None,
        train_transforms: th.Optional[th.Union[dict, th.Any]] = None,
        val_transforms: th.Optional[th.Union[dict, th.Any]] = None,
        test_transforms: th.Optional[th.Union[dict, th.Any]] = None,
        # batch_size
        batch_size: th.Optional[int] = 16,
        train_batch_size: th.Optional[int] = None,
        val_batch_size: th.Optional[int] = None,
        test_batch_size: th.Optional[int] = None,
        # pin memory
        pin_memory: bool = True,
        train_pin_memory: th.Optional[bool] = None,
        val_pin_memory: th.Optional[bool] = None,
        test_pin_memory: th.Optional[bool] = None,
        # shuffle
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
        # num_workers
        num_workers: int = 0,
        train_num_workers: th.Optional[int] = None,
        val_num_workers: th.Optional[int] = None,
        test_num_workers: th.Optional[int] = None,
        # seed
        seed: int = 0,
        # extra parameters (ignored)
        **kwargs,
    ):
        super().__init__()
        # transforms
        transforms = transforms if transforms is not None else ToTensor()
        self.transforms_train = initialize_transforms(train_transforms if train_transforms is not None else transforms)
        self.transforms_val = initialize_transforms(val_transforms if val_transforms is not None else transforms)
        self.transforms_test = initialize_transforms(test_transforms if test_transforms is not None else transforms)

        # seed
        self.seed = seed

        # data
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_dataset = dataset
        self.val_size = val_size
        assert (
            val_size is None or isinstance(val_size, int) or (isinstance(val_size, float) and val_size < 1)
        ), "invalid validation size is provided (either int, float (between zero and one) or None)"
        self.val_dataset = dataset if val_size > 0  else dataset
        self.test_dataset = test_dataset if test_dataset is not None else dataset
        self.train_dataset_args = {
            **(dataset_args or dict()),
            **(train_dataset_args or dict()),
        }
        self.val_dataset_args = {
            **(dataset_args or dict()),
            **(val_dataset_args or dict()),
        }
        self.test_dataset_args = {
            **(dataset_args or dict()),
            **(test_dataset_args or dict()),
        }

        # normal classes
        self.normal_targets = normal_targets

        # batch_size
        self.train_batch_size = train_batch_size if train_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        assert (
            self.train_batch_size or self.val_batch_size or self.test_batch_size
        ), "at least one of batch_sizes should be a positive number"

        # pin memory
        self.train_pin_memory = train_pin_memory if train_pin_memory is not None else pin_memory
        self.val_pin_memory = val_pin_memory if val_pin_memory is not None else pin_memory
        self.test_pin_memory = test_pin_memory if test_pin_memory is not None else pin_memory

        # shuffle
        self.train_shuffle, self.val_shuffle, self.test_shuffle = (
            train_shuffle,
            val_shuffle,
            test_shuffle,
        )

        # num_workers
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.test_num_workers = test_num_workers if test_num_workers is not None else num_workers

    @staticmethod
    def get_dataset(dataset: th.Union[str, Dataset], **params):
        if isinstance(dataset, Dataset):
            return dataset
        else:
            return get_value(dataset, strict=True)(**params)

    def setup(self, stage: th.Optional[str] = None) -> None:
        if (stage == "fit" or stage == "tune") and self.train_batch_size:
            dataset = self.get_dataset(
                self.train_dataset,
                transform=self.transforms_train,
                **self.train_dataset_args,
            )
            if not self.val_size:
                # setup train data (in case validation is not to be a subset of provided dataset with val_size)
                self.train_data = (
                    dataset if self.normal_targets is None else NormalDataset(dataset, self.normal_targets)
                )
            if self.val_size and self.val_batch_size:
                # split dataset into train and val in case val_size is provided
                train_len = (
                    len(dataset) - self.val_size
                    if isinstance(self.val_size, int)
                    else int(len(dataset) * (1 - self.val_size))
                )
                prng = torch.Generator()
                prng.manual_seed(self.seed)  # for reproducibility
                train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len], generator=prng)
                if self.normal_targets is not None:
                    self.train_data, self.val_data = NormalDataset(train_data, self.normal_targets), IsNormalDataset(
                        val_data, self.normal_targets
                    )
                else:
                    self.train_data, self.val_data = train_data, val_data
            elif self.val_batch_size:
                val_data = self.get_dataset(
                    self.val_dataset,
                    transform=self.transforms_val,
                    **self.val_dataset_args,
                )
                self.val_data = (
                    val_data if self.normal_targets is None else IsNormalDataset(val_data, self.normal_targets)
                )

        elif stage == "test" and self.test_batch_size:
            test_data = (
                self.get_dataset(
                    self.test_dataset,
                    transform=self.transforms_test,
                    **self.test_dataset_args,
                ),
            )
            self.test_data = (
                test_data if self.normal_targets is None else IsNormalDataset(test_data, self.normal_targets)
            )

    def get_dataloader(self, name, batch_size=None, shuffle=None, num_workers=None, pin_memory=None, **params):
        data = getattr(self, f"{name}_data")
        if data is None:
            return None
        batch_size = batch_size if batch_size is not None else getattr(self, f"{name}_batch_size")
        shuffle = shuffle if shuffle is not None else getattr(self, f"{name}_shuffle")
        num_workers = num_workers if num_workers is not None else getattr(self, f"{name}_num_workers")
        pin_memory = pin_memory if pin_memory is not None else getattr(self, f"{name}_pin_memory")
        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **params,
        )

    def train_dataloader(self, **params):
        return self.get_dataloader("train", **params)

    def val_dataloader(self, **params):
        return self.get_dataloader("val", **params)

    def test_dataloader(self, **params):
        return self.get_dataloader("test", **params)
