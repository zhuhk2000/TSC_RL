import os
import time
from dataclasses import dataclass




TIME_STAMP = int(time.time())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EnvCFG:
    file_dir: str = os.path.join(BASE_DIR, "data")
    render: bool = False
    label: str = 'env'
    delta_simu_steps: int = 5
    total_simu_steps: int = 3600
    num_seg: int = 10
    with_phase: bool = True


@dataclass
class ModelCFG:
    model_class: str = "SAC"
    label: str = 'model'
    actor_shape: tuple = (488, 512, 512, 512, 8)
    qnet_shape: tuple = (488, 512, 512, 512, 8)
    layer_norm: bool = True
    act: str = 'SiLU'
    device = DEVICE


@dataclass
class ReplayBufferCFG:
    maxlen: int = 100000
    start_sample_size: int = 2048
    data_reuse: int = 16
    priority_replay: bool = False
    batch_size: int = 1024
    # batch maker
    num_batch_maker: int = 5
    device = DEVICE


@dataclass
class LearnerCFG:
    lr: float = 1e-3
    gamma: float = 0.997
    # soft_update_decay: float = 0.98 
    alpha: float = 0.2
    epsilon_begin: float = 1
    epsilon_end: float = 0.01


@dataclass
class CFG:
    # logger
    logger_file_dir: str = os.path.join(BASE_DIR, f"logs/exp_{TIME_STAMP}")
    num_samplers: int = 64
    num_steps: int = 32
    # cache
    cached_intervals: int = 10
    saved_step_intervals: int = 10000
    # env
    env_cfg: EnvCFG = EnvCFG()
    model_cfg: ModelCFG = ModelCFG()
    replay_buffer_cfg: ReplayBufferCFG = ReplayBufferCFG()
    learner_cfg: LearnerCFG = LearnerCFG()