from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

# import omegaconf
import pytorch_lightning as pl
import torch
import tqdm
from hydra import compose, initialize
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

with initialize(
    version_base="1.3",
    config_path="../../configs/",
):
    cfg = compose(config_name="main")

IMAGE_SIZE = cfg.data.dataset.image_size
PADDING_SIZE = cfg.data.dataset.padding_size
VIEWS = cfg.data.dataset.views


class MultiViewGeoLocalizationDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        transform_dict: Dict[str, transforms.Compose],
        mode: str = "train",
    ):
        """
        Унифицированный датасет для работы с разными типами съемки

        Args:
            data_dir: Путь к корневой директории датасета
            transform_dict: Словарь трансформаций для каждого типа съемки
            mode: Режим работы ('train' или 'val')
        """
        self.data_dir = data_dir
        self.transform_dict = transform_dict
        self.mode = mode
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.view_counts = defaultdict(int)

        # Создаем аугментации
        self.transform_dict = transform_dict

        # Собираем данные
        self._load_data()

    def _load_data(self):
        """Загружает данные из всех типов съемки"""
        class_idx = 0
        for view in VIEWS:
            view_path = self.data_dir / "train" / view

            # Собираем данные для текущего типа съемки
            for building_path in tqdm.tqdm(view_path.iterdir(), f"processing {view}"):
                building_cls = building_path.name
                # Создаем ID класса, если он новый
                if building_cls not in self.class_to_idx.keys():
                    self.class_to_idx[building_cls] = class_idx
                    self.idx_to_class[class_idx] = building_cls
                    class_idx += 1

                cls_id = self.class_to_idx[building_cls]

                # Добавляем все изображения здания
                for img_path in building_path.iterdir():
                    self.samples.append((img_path, cls_id, view))
                    self.view_counts[view] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, view = self.samples[idx]

        # Загрузка изображения
        img = Image.open(img_path).convert("RGBA")
        img = transforms.ToTensor()(img)
        img = img[:3]
        img = transforms.ToPILImage()(img)

        # Применение трансформаций
        transform = self.transform_dict["val"] if self.mode == "val" else self.transform_dict[view]
        img = transform(img)

        return img, class_id, view


class GeoLocalizationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform_dict=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_dict = transform_dict
        if self.transform_dict is None:
            self.set_default_transform_dict()

    def prepare_data(self):
        # Проверяем доступность данных
        for view in VIEWS:
            view_path = self.data_dir / "train" / view
            if not view_path.exists():
                print(f"Warning: View directory not found - {view_path}")
                raise ValueError

    def setup(self, stage: Optional[str] = None):
        # Загружаем полный датасет
        full_dataset = MultiViewGeoLocalizationDataset(self.data_dir, self.transform_dict, mode="train")
        # Разделение на train/val
        all_indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=0.2,
            stratify=[full_dataset.samples[i][1] for i in all_indices],
            random_state=42,
        )
        # Создаем подмножества
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        # Для валидации устанавливаем единые трансформации
        for i in val_indices:
            _, _, view = full_dataset.samples[i]
            full_dataset.transform_dict[view] = self.transform_dict["val"]

    def set_default_transform_dict(self):
        train_transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.Pad(PADDING_SIZE, padding_mode="edge"),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        train_satelite_transform_list = [
            transforms.Resize(IMAGE_SIZE),
            transforms.Pad(PADDING_SIZE, padding_mode="edge"),
            transforms.RandomAffine(90),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        val_transform_list = [
            transforms.Resize(size=IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        self.transform_dict = {}

        for view in VIEWS:
            self.transform_dict[view] = (
                transforms.Compose(train_transform_list)
                if view != "satellite"
                else transforms.Compose(train_satelite_transform_list)
            )

        self.transform_dict["val"] = transforms.Compose(val_transform_list)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Важно для работы miner
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
