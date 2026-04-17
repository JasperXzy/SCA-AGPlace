from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyYAML is required when using --config YAML files.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if value[:1] in "[{":
        try:
            import yaml

            return yaml.safe_load(value)
        except Exception:
            return value
    if re.fullmatch(r"[+-]?(0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(r"[+-]?([0-9]*\.[0-9]+|[0-9]+\.[0-9]*)([eE][+-]?[0-9]+)?", value):
        return float(value)
    if re.fullmatch(r"[+-]?[1-9][0-9]*[eE][+-]?[0-9]+", value):
        return float(value)
    return value


def _field_names(obj: Any) -> set[str]:
    return {item.name for item in fields(obj)}


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            lower = value.lower()
            if lower in {"true", "false"}:
                return lower == "true"
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool) and value is not None:
        return int(value)
    if isinstance(current, float) and value is not None:
        return float(value)
    if isinstance(current, list):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return [_parse_scalar(item) for item in value.split(",")]
    if isinstance(current, str) and value is not None:
        return str(value)
    return copy.deepcopy(value)


def _split_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split("_") if part]
    return list(value)


def _expand_user(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return os.path.expanduser(value)
    return value


@dataclass
class TrainerCfg:
    accelerator: Optional[str] = None
    devices: Any = None
    strategy: Any = None
    precision: Any = None
    reload_dataloaders_every_n_epochs: Optional[int] = None
    accumulate_grad_batches: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    num_sanity_val_steps: Optional[int] = None
    enable_model_summary: Optional[bool] = None
    check_val_every_n_epoch: Optional[int] = None
    max_epochs: Optional[int] = None
    default_root_dir: Optional[str] = None
    sync_batchnorm: Optional[bool] = None
    enable_progress_bar: Optional[bool] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    callbacks: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        data = {
            key: value
            for key, value in asdict(self).items()
            if key != "extra" and value is not None
        }
        data.update(self.extra)
        return data


@dataclass
class ModelCfg:
    features_dim: int = 256
    lr: float = 1e-5
    lrpc: float = 1e-4
    lrdb: float = 1e-5
    lrdino: float = 0.0
    unfreeze_dino_mode: str = "frozen"
    dino_extract_blocks: str = "7_15_23"
    utonia_pretrained: str = "utonia"
    unfreeze_utonia_mode: str = "last1"
    lrutonia: float = 1e-5
    utonia_extract_stages: str = "0_2_4"
    amp_dtype: str = "none"
    share_db: bool = False
    share_dbfe: bool = False
    mm_imgfe_dim: int = 1024
    mm_voxfe_planes: str = "64_128_256"
    mm_voxfe_dim: int = 256
    mm_bevfe_planes: str = "64_128_256"
    mm_bevfe_dim: int = 256
    mm_stg2fuse_dim: int = 256
    output_type: Any = "image_vox_shallow"
    output_l2: bool = True
    final_type: Any = "imageorg_voxorg_shalloworg_stg2image_stg2vox"
    final_fusetype: str = "add"
    final_l2: bool = False
    image_embed: str = "stg2image"
    cloud_embed: str = "stg2vox"
    image_weight: float = 1.0
    image_learnweight: bool = False
    vox_weight: float = 1.0
    vox_learnweight: bool = False
    shallow_weight: float = 1.0
    shallow_learnweight: bool = False
    diff_type: str = "fcode@relu"
    diff_direction: str = "backward"
    odeint_method: str = "euler"
    odeint_size: float = 0.1
    tol: float = 1e-3
    imagevoxorg_weight: float = 0.0
    imagevoxorg_learnweight: bool = False
    shalloworg_weight: float = 1.0
    shalloworg_learnweight: bool = False
    stg2imagevox_weight: float = 0.1
    stg2imagevox_learnweight: bool = False
    stg2fuse_weight: float = 0.0
    stg2fuse_learnweight: bool = False
    stg2nlayers: int = 1
    stg2fuse_type: Optional[str] = "basic"
    stg2_type: str = "full"
    stg2_useproj: bool = True


@dataclass
class DataCfg:
    cuda: str = "0"
    device: str = "cuda"
    num_workers: int = 8
    machine: str = "5080"
    dataset: str = "kitti360"
    datasets_folder: str = ""
    dataset_name: str = ""
    dataroot: Optional[str] = None
    maptype: str = "satellite"
    traindownsample: int = 4
    train_ratio: float = 0.85
    camnames: str = "00"
    train_batch_size: int = 16
    infer_batch_size: int = 32
    cache_refresh_rate: int = 4000
    queries_per_epoch: int = 16000
    val_positive_dist_threshold: int = 25
    train_positives_dist_threshold: int = 10
    neg_samples_num: int = 1000
    negs_num_per_query: int = 10
    epochs_num: int = 100
    resize: List[int] = field(default_factory=lambda: [256, 256])
    color_jitter: float = 0.0
    quant_size: float = 2.0
    db_cropsize: int = 256
    db_resize: int = 224
    db_jitter: float = 0.0
    q_resize: int = 224
    q_jitter: float = 0.0
    read_pc: bool = True
    test_method: str = "hard_resize"
    majority_weight: float = 0.01
    efficient_ram_testing: bool = False
    recall_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    rand_perspective: float = 0.0
    horizontal_flip: bool = False
    random_resized_crop: float = 0.0
    random_rotation: float = 0.0
    disable_dataset_tqdm: bool = False
    bev_cropsize: int = 256
    bev_resize: float = 1.0
    bev_resize_mode: str = "nearest"
    bev_rotate: float = 0.0
    bev_rotate_mode: str = "nearest"
    bev_jitter: float = 0.0
    bev_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    bev_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    sph_resize: float = 1.0
    sph_jitter: float = 0.0
    sph_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    sph_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class LossCfg:
    otherloss_type: str = "bce"
    otherloss_weight: float = 0.01
    tripletloss_weight: float = 1.0
    patience: int = 50
    margin: float = 0.1
    mining: str = "partial_sep"


@dataclass
class LoggingCfg:
    csv: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    litlogger: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    seed: int = 0
    resume: Optional[str] = None
    save_dir: str = "default"
    exp_name: str = "none"
    ckpt_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    _SECTION_ORDER = ("data", "model", "loss", "logging", "trainer")
    _LIST_CLI_FIELDS = {
        "resize",
        "recall_values",
        "bev_mean",
        "bev_std",
        "sph_mean",
        "sph_std",
    }

    @classmethod
    def from_argv(cls, argv: Optional[Iterable[str]] = None) -> Tuple["Config", str]:
        tokens = list(argv or [])
        command = "fit"
        if tokens and not tokens[0].startswith("-"):
            command = tokens.pop(0)
            if command == "train":
                command = "fit"

        config_paths: List[str] = []
        overrides: Dict[str, Any] = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--config":
                config_paths.append(tokens[index + 1])
                index += 2
                continue
            if token.startswith("--config="):
                config_paths.append(token.split("=", 1)[1])
                index += 1
                continue
            if not token.startswith("--"):
                raise ValueError(f"Unexpected positional argument: {token}")

            key = token[2:]
            if "=" in key:
                key, raw_value = key.split("=", 1)
                value = _parse_scalar(raw_value)
                index += 1
            elif cls._expects_list(key):
                values = []
                index += 1
                while index < len(tokens) and not tokens[index].startswith("--"):
                    values.append(_parse_scalar(tokens[index]))
                    index += 1
                value = values
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = _parse_scalar(tokens[index + 1])
                index += 2
            else:
                value = True
                index += 1
            _set_nested(overrides, key, value)

        merged: Dict[str, Any] = {}
        for path in config_paths:
            _deep_update(merged, _load_yaml(path))
        _deep_update(merged, overrides)
        return cls.from_dict(merged), command

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Config":
        cfg = cls()
        cfg.update(values)
        return cfg.resolve()

    @classmethod
    def _expects_list(cls, key: str) -> bool:
        return key.split(".")[-1] in cls._LIST_CLI_FIELDS

    def update(self, values: Mapping[str, Any]) -> None:
        values = copy.deepcopy(dict(values))
        for section_name in self._SECTION_ORDER:
            section_values = values.pop(section_name, None)
            if section_values is not None:
                if not isinstance(section_values, Mapping):
                    raise TypeError(f"{section_name} config must be a mapping")
                self._apply_section(section_name, section_values)

        for key, value in values.items():
            self._set_known_field(key, value)

    def resolve(self) -> "Config":
        self._resolve_machine_paths()
        self._normalise_values()
        self._resolve_experiment_name()
        self._validate()
        return self

    def apply_runtime_environment(self) -> None:
        if "RANK" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda)

    def default_precision(self) -> str:
        if self.amp_dtype == "bf16":
            return "bf16-mixed"
        if self.amp_dtype == "fp16":
            return "16-mixed"
        return "32-true"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        flat = {
            "seed": self.seed,
            "resume": self.resume,
            "save_dir": self.save_dir,
            "exp_name": self.exp_name,
            "ckpt_path": self.ckpt_path,
        }
        for section_name in ("model", "data", "loss"):
            for key, value in asdict(getattr(self, section_name)).items():
                flat[key] = value
        return flat

    def __getattr__(self, name: str) -> Any:
        for section_name in ("data", "model", "loss", "logging", "trainer"):
            section = self.__dict__.get(section_name)
            if section is not None and name in _field_names(section):
                return getattr(section, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        root_fields = {item.name for item in fields(type(self))} if hasattr(type(self), "__dataclass_fields__") else set()
        if name.startswith("_") or name in root_fields:
            object.__setattr__(self, name, value)
            return
        for section_name in ("data", "model", "loss", "logging", "trainer"):
            section = self.__dict__.get(section_name)
            if section is not None and name in _field_names(section):
                setattr(section, name, _coerce_value(getattr(section, name), value))
                return
        object.__setattr__(self, name, value)

    def _apply_section(self, section_name: str, values: Mapping[str, Any]) -> None:
        if section_name == "trainer":
            self._update_dataclass(self.trainer, values, allow_extra=True)
            return
        if section_name == "logging":
            for key, value in values.items():
                if key in {"csv", "litlogger"} and isinstance(value, Mapping):
                    target = getattr(self.logging, key)
                    _deep_update(target, value)
                elif key in _field_names(self.logging):
                    setattr(self.logging, key, _coerce_value(getattr(self.logging, key), value))
                else:
                    self.logging.extra[key] = copy.deepcopy(value)
            return

        for key, value in values.items():
            self._set_known_field(key, value, preferred=section_name)

    def _set_known_field(self, key: str, value: Any, preferred: Optional[str] = None) -> None:
        root_fields = _field_names(self)
        if key in root_fields and key not in self._SECTION_ORDER:
            setattr(self, key, _coerce_value(getattr(self, key), value))
            return

        ordered_sections = []
        if preferred is not None:
            ordered_sections.append(preferred)
        ordered_sections.extend(
            section for section in ("data", "model", "loss") if section not in ordered_sections
        )
        for section_name in ordered_sections:
            section = getattr(self, section_name)
            if key in _field_names(section):
                setattr(section, key, _coerce_value(getattr(section, key), value))
                return
        self.extra[key] = copy.deepcopy(value)

    @staticmethod
    def _update_dataclass(target: Any, values: Mapping[str, Any], allow_extra: bool = False) -> None:
        names = _field_names(target)
        for key, value in values.items():
            if key in names and key != "extra":
                setattr(target, key, _coerce_value(getattr(target, key), value))
            elif allow_extra:
                target.extra[key] = copy.deepcopy(value)
            else:
                raise KeyError(f"Unknown config key: {key}")

    def _resolve_machine_paths(self) -> None:
        machine_paths = {
            "4090": {
                "kitti360": "/mnt/sda/ZhengyiXu/datasets/cmvpr/kitti360/KITTI-360",
                "nuscenes": "/mnt/sda/ZhengyiXu/datasets/radar/nuscenes",
            },
            "5080": {
                "kitti360": "/home/jasperxzy/Datasets/cmvpr/kitti360/KITTI-360",
                "nuscenes": "/home/jasperxzy/Datasets/radar/nuscenes",
            },
        }
        default_root = machine_paths.get(self.data.machine, {}).get(self.data.dataset)
        if default_root and not self.data.dataroot:
            self.data.dataroot = default_root
        if self.data.dataset and not self.data.dataset_name:
            self.data.dataset_name = self.data.dataset

    def _normalise_values(self) -> None:
        self.model.output_type = _split_tokens(self.model.output_type)
        self.model.final_type = _split_tokens(self.model.final_type)
        self.data.dataroot = _expand_user(self.data.dataroot)
        self.data.datasets_folder = _expand_user(self.data.datasets_folder)
        self.model.utonia_pretrained = _expand_user(self.model.utonia_pretrained)
        self.resume = _expand_user(self.resume)
        self.save_dir = _expand_user(self.save_dir)

    def _resolve_experiment_name(self) -> None:
        if self.exp_name in {None, "", "none"}:
            self.exp_name = (
                f"{self.seed}_"
                f"ep{self.epochs_num}"
                f"_{self.dataset}"
                f"_{self.camnames}"
                f"_{self.cache_refresh_rate}"
                f"_{self.queries_per_epoch}"
                f"_{self.maptype}"
                f"_trbs{self.train_batch_size}"
                f"_{self.infer_batch_size}"
                f"_{self.traindownsample}"
                f"_{self.train_ratio}"
                f"_pc{self.read_pc}"
            )
        if self.save_dir == "default":
            self.save_dir = os.path.join("logs", self.exp_name)

    def _validate(self) -> None:
        if self.queries_per_epoch % self.cache_refresh_rate != 0:
            raise ValueError(
                "Ensure that queries_per_epoch is divisible by cache_refresh_rate, "
                f"because {self.queries_per_epoch} is not divisible by {self.cache_refresh_rate}"
            )
        if self.mining == "msls_weighted" and self.dataset_name != "msls":
            raise ValueError(
                "msls_weighted mining can only be applied to msls dataset, "
                f"but you're using it on {self.dataset_name}"
            )
