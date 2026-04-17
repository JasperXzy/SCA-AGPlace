from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml
except ImportError:  # pragma: no cover - depends on environment
    yaml = None


def _load_yaml(path: str) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover - depends on environment
        raise ImportError("PyYAML is required when using --config YAML files.")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
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
        if yaml is None:
            return value
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
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


def _split_token_string(value: str, separator: str = "_") -> list[str]:
    delimiter = "," if "," in value else separator
    return [part.strip() for part in value.split(delimiter) if part.strip()]


def _normalise_token_list(value: Any, field_name: str, item_type: type = str, separator: str = "_") -> list[Any]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_items: list[Any] = _split_token_string(value, separator=separator)
    else:
        raw_items = []
        for item in value:
            if isinstance(item, str) and (separator in item or "," in item):
                raw_items.extend(_split_token_string(item, separator=separator))
            else:
                raw_items.append(item)

    return [_coerce_token_item(item, field_name, item_type) for item in raw_items]


def _coerce_token_item(item: Any, field_name: str, item_type: type) -> Any:
    try:
        return item_type(item)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a list of {item_type.__name__} values; got {item!r}") from exc


def _normalise_optional_token_list(value: Any, field_name: str, item_type: type = str) -> list[Any] | None:
    if value is None:
        return None
    return _normalise_token_list(value, field_name=field_name, item_type=item_type)


def _validate_choices(field_name: str, values: Iterable[str], choices: set[str]) -> None:
    unknown = [value for value in values if value not in choices]
    if unknown:
        raise ValueError(f"{field_name} has unsupported values {unknown}; expected one of {sorted(choices)}")


def _join_tokens(values: Any) -> str:
    if isinstance(values, str):
        return values
    return "_".join(str(value) for value in values)


def _expand_user(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return os.path.expanduser(value)
    return value


def _machine_config_path(machine: str) -> Path:
    config_dir = os.environ.get("MAG_VLAQ_MACHINE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / f"{machine}.yaml"
    return Path(__file__).resolve().parents[2] / "configs" / "machines" / f"{machine}.yaml"


def _load_machine_paths(machine: str) -> dict[str, Any]:
    machine_config = _machine_config_path(machine)
    if not machine_config.exists():
        return {}

    values = _load_yaml(str(machine_config))
    paths = values.get("paths", values)
    if not isinstance(paths, Mapping):
        raise TypeError(f"Machine config {machine_config} must be a mapping or contain a 'paths' mapping")
    return {str(dataset): _expand_user(path) for dataset, path in paths.items() if path}


@dataclass
class TrainerCfg:
    accelerator: str | None = None
    devices: Any = None
    strategy: Any = None
    precision: Any = None
    reload_dataloaders_every_n_epochs: int | None = None
    accumulate_grad_batches: int | None = None
    log_every_n_steps: int | None = None
    num_sanity_val_steps: int | None = None
    enable_model_summary: bool | None = None
    check_val_every_n_epoch: int | None = None
    max_epochs: int | None = None
    default_root_dir: str | None = None
    sync_batchnorm: bool | None = None
    enable_progress_bar: bool | None = None
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: str | None = None
    callbacks: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        data = {key: value for key, value in asdict(self).items() if key != "extra" and value is not None}
        data.update(self.extra)
        return data


@dataclass
class ModelCfg:
    features_dim: int = 512
    lr: float = 1e-5
    lrpc: float = 1e-4
    lrdb: float = 1e-5
    lrdino: float = 1e-6
    unfreeze_dino_mode: str = "last2"
    dino_extract_blocks: list[int] = field(default_factory=lambda: [7, 15, 23])
    utonia_pretrained: str = "utonia"
    unfreeze_utonia_mode: str = "last1"
    lrutonia: float = 1e-6
    utonia_extract_stages: list[int] = field(default_factory=lambda: [2, 3, 4])
    amp_dtype: str = "bf16-mixed"
    share_db: bool = False
    share_dbfe: bool = False
    mm_imgfe_dim: int = 1024
    mm_voxfe_planes: list[int] = field(default_factory=lambda: [512, 512, 512])
    mm_voxfe_dim: int = 512
    mm_bevfe_planes: list[int] = field(default_factory=lambda: [64, 128, 256])
    mm_bevfe_dim: int = 256
    mm_stg2fuse_dim: int = 512
    output_type: list[str] = field(default_factory=lambda: ["image", "vox", "shallow"])
    output_l2: bool = True
    final_type: list[str] = field(default_factory=lambda: ["imageorg", "voxorg", "shalloworg", "stg2image", "stg2vox"])
    final_fusetype: str = "add"
    final_l2: bool = False
    image_embed: str = "stg2image"
    cloud_embed: str = "stg2vox"
    image_weight: float = 1.0
    image_learnweight: bool = True
    vox_weight: float = 1.0
    vox_learnweight: bool = True
    shallow_weight: float = 1.0
    shallow_learnweight: bool = True
    diff_type: list[str] = field(default_factory=lambda: ["fcode@relu"])
    diff_direction: str = "backward"
    odeint_method: str = "euler"
    odeint_size: float = 0.1
    tol: float = 1e-3
    imagevoxorg_weight: float = 1.0
    imagevoxorg_learnweight: bool = True
    shalloworg_weight: float = 1.0
    shalloworg_learnweight: bool = True
    stg2imagevox_weight: float = 0.1
    stg2imagevox_learnweight: bool = True
    stg2fuse_weight: float = 0.1
    stg2fuse_learnweight: bool = True
    stg2nlayers: int = 1
    stg2fuse_type: list[str] | None = field(default_factory=lambda: ["basic"])
    stg2_type: str = "full"
    stg2_useproj: bool = True


@dataclass
class DataCfg:
    cuda: str = "0"
    device: str = "cuda"
    num_workers: int = 8
    cache_num_workers: int = 0
    worker_multiprocessing_context: str | None = "spawn"
    omp_num_threads: int = 16
    machine: str = "5080"
    dataset: str = "kitti360"
    datasets_folder: str = ""
    dataset_name: str = ""
    dataroot: str | None = None
    maptype: list[str] = field(default_factory=lambda: ["satellite"])
    traindownsample: int = 4
    train_ratio: float = 0.85
    camnames: list[str] = field(default_factory=lambda: ["00"])
    train_batch_size: int = 16
    infer_batch_size: int = 32
    cache_refresh_rate: int = 4000
    queries_per_epoch: int = 16000
    val_positive_dist_threshold: int = 25
    train_positives_dist_threshold: int = 10
    neg_samples_num: int = 1000
    negs_num_per_query: int = 10
    epochs_num: int = 100
    resize: list[int] = field(default_factory=lambda: [256, 256])
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
    recall_values: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
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
    bev_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    bev_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    sph_resize: float = 1.0
    sph_jitter: float = 0.0
    sph_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    sph_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


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
    csv: dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    litlogger: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    seed: int = 0
    resume: str | None = None
    save_dir: str = "default"
    exp_name: str = "none"
    ckpt_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _SECTION_ORDER: ClassVar[tuple[str, ...]] = ("data", "model", "loss", "logging", "trainer")
    _LIST_CLI_FIELDS: ClassVar[set[str]] = {
        "camnames",
        "diff_type",
        "dino_extract_blocks",
        "final_type",
        "maptype",
        "mm_bevfe_planes",
        "mm_voxfe_planes",
        "output_type",
        "resize",
        "recall_values",
        "stg2fuse_type",
        "bev_mean",
        "bev_std",
        "sph_mean",
        "sph_std",
        "utonia_extract_stages",
    }

    @classmethod
    def from_argv(cls, argv: Iterable[str] | None = None) -> tuple[Config, str]:
        tokens = list(argv or [])
        command = "fit"
        if tokens and not tokens[0].startswith("-"):
            command = tokens.pop(0)
            if command == "train":
                command = "fit"

        config_paths: list[str] = []
        overrides: dict[str, Any] = {}
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

        merged: dict[str, Any] = {}
        for path in config_paths:
            _deep_update(merged, _load_yaml(path))
        _deep_update(merged, overrides)
        return cls.from_dict(merged), command

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Config:
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

    def resolve(self) -> Config:
        self._resolve_machine_paths()
        self._normalise_values()
        self._resolve_experiment_name()
        self._validate()
        return self

    def apply_runtime_environment(self) -> None:
        if "RANK" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda)
        os.environ.setdefault("OMP_NUM_THREADS", str(self.omp_num_threads))

    def default_precision(self) -> str:
        if self.amp_dtype == "bf16":
            return "bf16-mixed"
        if self.amp_dtype == "fp16":
            return "16-mixed"
        return "32-true"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        flat = {
            "seed": self.seed,
            "resume": self.resume,
            "save_dir": self.save_dir,
            "exp_name": self.exp_name,
            "ckpt_path": self.ckpt_path,
        }
        for section_name in ("model", "data", "loss"):
            flat.update({key: value for key, value in asdict(getattr(self, section_name)).items()})
        return flat

    def __getattr__(self, name: str) -> Any:
        for section_name in ("data", "model", "loss", "logging", "trainer"):
            section = self.__dict__.get(section_name)
            if section is not None and name in _field_names(section):
                return getattr(section, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        root_fields = (
            {item.name for item in fields(type(self))} if hasattr(type(self), "__dataclass_fields__") else set()
        )
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

    def _set_known_field(self, key: str, value: Any, preferred: str | None = None) -> None:
        root_fields = _field_names(self)
        if key in root_fields and key not in self._SECTION_ORDER:
            setattr(self, key, _coerce_value(getattr(self, key), value))
            return

        ordered_sections = []
        if preferred is not None:
            ordered_sections.append(preferred)
        ordered_sections.extend(section for section in ("data", "model", "loss") if section not in ordered_sections)
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
        machine_paths = _load_machine_paths(str(self.data.machine))
        default_root = machine_paths.get(self.data.dataset)
        if default_root and not self.data.dataroot:
            self.data.dataroot = default_root
        if self.data.dataset and not self.data.dataset_name:
            self.data.dataset_name = self.data.dataset

    def _normalise_values(self) -> None:
        self.model.dino_extract_blocks = _normalise_token_list(
            self.model.dino_extract_blocks, "model.dino_extract_blocks", int
        )
        self.model.utonia_extract_stages = _normalise_token_list(
            self.model.utonia_extract_stages, "model.utonia_extract_stages", int
        )
        self.model.mm_voxfe_planes = _normalise_token_list(self.model.mm_voxfe_planes, "model.mm_voxfe_planes", int)
        self.model.mm_bevfe_planes = _normalise_token_list(self.model.mm_bevfe_planes, "model.mm_bevfe_planes", int)
        self.model.output_type = _normalise_token_list(self.model.output_type, "model.output_type")
        self.model.final_type = _normalise_token_list(self.model.final_type, "model.final_type")
        self.model.diff_type = _normalise_token_list(self.model.diff_type, "model.diff_type")
        self.model.stg2fuse_type = _normalise_optional_token_list(self.model.stg2fuse_type, "model.stg2fuse_type")
        self.data.maptype = _normalise_token_list(self.data.maptype, "data.maptype")
        self.data.camnames = _normalise_token_list(self.data.camnames, "data.camnames")
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
                f"_{_join_tokens(self.camnames)}"
                f"_{self.cache_refresh_rate}"
                f"_{self.queries_per_epoch}"
                f"_{_join_tokens(self.maptype)}"
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
                f"msls_weighted mining can only be applied to msls dataset, but you're using it on {self.dataset_name}"
            )
        _validate_choices("model.output_type", self.model.output_type, {"image", "vox", "shallow", "addorg"})
        _validate_choices(
            "model.final_type",
            self.model.final_type,
            {"imageorg", "voxorg", "shalloworg", "stg2image", "stg2vox", "stg2fuse"},
        )
        _validate_choices("data.maptype", self.data.maptype, {"satellite", "roadmap"})
        _validate_choices("data.camnames", self.data.camnames, {"00", "0203", "f", "fl", "fr", "b", "bl", "br"})
        if self.model.stg2fuse_type is not None:
            _validate_choices("model.stg2fuse_type", self.model.stg2fuse_type, {"basic"})
        if len(self.model.dino_extract_blocks) != len(self.model.mm_voxfe_planes):
            raise ValueError(
                "model.dino_extract_blocks and model.mm_voxfe_planes must have the same length, "
                f"got {len(self.model.dino_extract_blocks)} and {len(self.model.mm_voxfe_planes)}"
            )
        if self.model.dino_extract_blocks != sorted(self.model.dino_extract_blocks):
            raise ValueError(
                "model.dino_extract_blocks must be in increasing block order, "
                f"got {self.model.dino_extract_blocks}"
            )
        if any(block < 0 or block > 23 for block in self.model.dino_extract_blocks):
            raise ValueError(
                "model.dino_extract_blocks must refer to DINOv2 ViT-L/14 block indexes in [0, 23], "
                f"got {self.model.dino_extract_blocks}"
            )
        if len(self.model.utonia_extract_stages) != len(self.model.mm_voxfe_planes):
            raise ValueError(
                "model.utonia_extract_stages and model.mm_voxfe_planes must have the same length, "
                f"got {len(self.model.utonia_extract_stages)} and {len(self.model.mm_voxfe_planes)}"
            )
        if any(dim != self.model.mm_stg2fuse_dim for dim in self.model.mm_voxfe_planes):
            raise ValueError(
                "model.mm_voxfe_planes must already match model.mm_stg2fuse_dim because "
                "FuseBlockToShallow no longer applies a second voxel projection, "
                f"got mm_voxfe_planes={self.model.mm_voxfe_planes} and "
                f"mm_stg2fuse_dim={self.model.mm_stg2fuse_dim}"
            )
        if self.model.mm_voxfe_dim != self.model.mm_voxfe_planes[-1]:
            raise ValueError(
                "model.mm_voxfe_dim must match the last projected Utonia stage dimension, "
                f"got mm_voxfe_dim={self.model.mm_voxfe_dim} and "
                f"mm_voxfe_planes[-1]={self.model.mm_voxfe_planes[-1]}"
            )
