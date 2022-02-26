import json

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
import torch.cuda
import torch.profiler

def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    if cfg.trainer.enable_checkpointing:
        checkpoint_callback = FileCheckpointHandler(
            cfg=cfg.checkpoint
        )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=cfg.trainer.gpus > 1
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    # Rajesh - Activate these lines for PyTorch Profiler
    #profiler_kwargs = {"profile_memory":True,"use_cuda":True, "record_shapes":True, "with_stack":True} #torch.profiler.profile.schedule(wait=1, warmup=1, active=3, repeat=2), "record_shapes":True}
    #profiler = PyTorchProfiler(dirpath = "/home/cc/data", filename = "./bs-64-fp32", export_to_chrome = True, **profiler_kwargs) 
    #profiler = SimpleProfiler(dirpath = "/home/cc/pytorch_lightning_simple",filename = "./bs64-fp32-inference")

    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
    #    profiler = profiler,
        callbacks=[checkpoint_callback] if cfg.trainer.enable_checkpointing else None,
    )
    
    torch.cuda.cudart().cudaProfilerStart()
    trainer.fit(model, data_loader)
    torch.cuda.cudart().cudaProfilerStop()
