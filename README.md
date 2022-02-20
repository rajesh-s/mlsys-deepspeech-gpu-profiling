# Profiling DeepSpeech on GPU systems 

- [1. Objective](#1-objective)
- [2. DeepSpeech PyTorch implementation from here](#2-deepspeech-pytorch-implementation-from-here)
- [3. Profiling](#3-profiling)
- [4. Variations](#4-variations)
- [5. Misc](#5-misc)
- [6. Results](#6-results)
  - [6.1. Task 1](#61-task-1)
  - [6.2. Task 2](#62-task-2)
  - [6.3. Task 3](#63-task-3)
  - [6.4. Task 4](#64-task-4)

## 1. Objective

Some helpful links that describe nvprof profiler include
https://github.com/mit-satori/getting-started/blob/master/tutorial-examples/nvprof-profiling/Satori_NVProf_Intro.pdf 
https://developer.nvidia.com/blog/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086

You are also welcome to use other profiling tools such as PyProf https://github.com/NVIDIA/PyProf and the PyTorch profiler https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/

GPU architecture details can be found at
https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
Tasks

There are four main profiling tasks that you will need to perform for this assignment. For each task you can include figures or data from multiple profiling tools

- Task 1, ML Model Bottlenecks: Given the default training setup in your ML model, which kernels (and correspondingly ML model layers) take the longest fraction of time? What is the GPU utilization for these kernels? You can quantify **utilization using occupancy** for each kernel.  For kernels which take the longest time, what are their bottlenecks? Discuss if it is related to compute / memory using relevant data from profiling.
- Task 2, Role of batch size: **Vary the batch size** from 1 to maximum batch size (in steps) supported by your GPU. What do you find changes in terms of utilization and bottlenecks?
- Task 3, Role of **quantization**: Vary the number of bits used by the model parameters (try FP 16, FP 32, FP 64). What do you find changes in terms of utilization and bottlenecks? You can use a fixed (default) batch size for this part
- Task 4, **Forward vs. Backward pass**: Change the ML model code to only perform the forward pass and compare the utilization / bottlenecks to when both forward and backward passes are performed.

Summarize your observations across all four tasks

## 2. DeepSpeech PyTorch implementation from [here](https://github.com/SeanNaren/deepspeech.pytorch)

- Pytorch highly recommends installing an Anaconda environment. Also, it simplifies the whole process. Thought of moving from virtualenv to conda

```bash

cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name deepspech_pytorch_env python=3
conda activate deepspech_pytorch_env
```

- Install PyTorch for CUDA 11.3 (other available version is 10.2) ![PyTorch config](images/2022-02-18-06-15-44.png)
- Installed pre-reqs inside the same environment for deepspech_pytorch_env
- Install nvidia-docker from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Install cuDNN. Full details [here](https://tikoehle.github.io/pytorch_conda_jupyterhub/nvidia_cuDNN.html). Be sure to ```sudo ldconfig``

```bash
   sudo cp include/cudnn.h /usr/local/cuda/include
   sudo cp lib64/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

- To download only specific dataset for training ```cd data && python librispeech.py --files-to-use="train-clean-100.tar.gz:dev-clean.tar.gz:test-clean.tar.gz:test-other.tar.gz"```
- Start training ```python train.py +configs=librispeech```
- Training working on GPU ![1](images/Screenshot%20from%202022-02-18%2014-01-48.png) ![2](images/Screenshot%20from%202022-02-18%2014-03-30.png)
- Nvprof warning ![3](images/Screenshot%20from%202022-02-18%2014-12-23.png)

## 3. Profiling

sudo -E env PATH=$PATH nvprof --metrics achieved_occupancy,sm_efficiency,sm_efficiency_instance,cf_executed,dram_utilization,dram_read_throughput,flop_count_dp,dram_write_throughput,flop_count_sp,gld_efficiency,gst_efficiency,ipc,ipc_instance,l2_utilization --log-file task1_metrics python train.py +configs=librispeech

```sudo -E env PATH=$PATH nvprof --metrics achieved_occupancy,sm_efficiency,cf_executed,ipc --log-file task1_metrics python train.py +configs=librispeech```

- Nvprof
  - normal: Time spent on each call ```sudo -E env PATH=$PATH nvprof python train.py +configs=librispeech```
    - metrics: ``` sudo -E env PATH=$PATH nvprof --metrics achieved_occupancy,sm_efficiency,sm_efficiency_instance,cf_executed,dram_utilization,dram_read_throughput,flop_count_dp,dram_write_throughput,flop_count_sp,gld_efficiency,gst_efficiency,ipc,ipc_instance,l2_utilization --log-file task1_metrics python train.py +configs=librispeech```
  - trace: Kernel launch parameter
  - nvvp: Images use analysis metrics only on top 5 kernels
- cProfile: Call trace esp when using framework where trainer.fit() was hiding all steps. Useful to track down epoch/steps
- PyTorch: Simple -> Forward vs Backward. Each stage in training
- cudaStart/Stop -> This works. About 25% savings in nvprof space
- PyTorchProfiler(): ```sudo -E env PATH=$PATH python train.py +configs=librispeech```
  - Instrumentation for Pytorch Profiler:
    ```py
    profiler = PyTorchProfiler(dirpath = "/home/cc/data", filename = "./bs-16", export_to_chrome = True, {"profile_memory":True, "schedule":torch.profiler.profile.schedule(wait=1, warmup=1, active=3, repeat=2)})
    ```

**TBD**:

- DLProf https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/

## 4. Variations

- Batch size: 1-256
- Precision: 16/32/64 flop_count_dp,flop_count_sp,
- Use inference

## 5. Misc

- ```sudo -E PATH=$PATH``` to pass environment variables under sudo for NVProf
- Run this to avoid conda auto-activate ```conda config --set auto_activate_base false```
- Get torch version inside python ```import torch print(torch.__version__)```
- Focussed profiling on CUDA [reference](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59) [2](https://gist.github.com/mcarilli/213a4e698e4a0ae2234ddee56f4f3f95)

## 6. Results

### 6.1. Task 1

1. ```sudo -E env PATH=$PATH nvprof --profile-from-start off --log-file Task1/nvprof_summary python train.py +configs=librispeech```
2. ```sudo -E env PATH=$PATH nvprof --profile-from-start off --kernels "maxwell_fp16_sgemm_fp16_128x32_nn:maxwell_fp16_sgemm_fp16_32x128_nt:maxwell_fp16_sgemm_fp16_32x128_nn:wgrad_alg0_engine:vectorized_elementwise_kernel:maxwell_fp16_sgemm_fp16_64x64_nn:elemWiseRNNcell:LSTM_elementWise_bp1:gemmk1_kernel:CUDA memcpy DtoD:CUDA memcpy HtoD:CUDA memcpy DtoH" --log-file Task1/nvprof_metrics --metrics achieved_occupancy,sm_efficiency,cf_executed,ipc python train.py +configs=librispeech```

--kernels "maxwell_fp16_sgemm_fp16_128x32_nn:maxwell_fp16_sgemm_fp16_32x128_nt:maxwell_fp16_sgemm_fp16_32x128_nn:wgrad_alg0_engine:vectorized_elementwise_kernel:maxwell_fp16_sgemm_fp16_64x64_nn:elemWiseRNNcell:LSTM_elementWise_bp1:gemmk1_kernel:memcpy"

### 6.2. Task 2

### 6.3. Task 3

### 6.4. Task 4