<h1 align="center">TrajBridge: Future-Constrained Conditional Flow Bridge for Multi-Agent Trajectory Reconstruction in Dense Traffic</h1>

<p align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=for-the-badge&amp;logo=pytorch" alt="PyTorch 2.1.0">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8-blue.svg?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python 3.8">
  </a>
  <a href="https://developer.nvidia.com/cuda-toolkit">
    <img src="https://img.shields.io/badge/CUDA-12.1-76B900.svg?style=for-the-badge&amp;logo=nvidia&amp;logoColor=white" alt="CUDA 12.1">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-7B42BC.svg?style=for-the-badge" alt="License: Apache 2.0">
  </a>
</p>

<p align="center">
  <img src="assets/overview.png" alt="TrajBridge architecture: map and history scene encoding; factorized agent, temporal and mode interaction; future-tail encoding; and conditional flow refinement of trajectory priors." width="100%">
</p>


## ✨ Highlights

<table>
  <tr>
    <td width="50%"><strong>↔️ Two-sided reconstruction</strong><br>Observed history and a measured future tail constrain the missing interval from both sides.</td>
    <td width="50%"><strong>🚘 Interaction-aware motion priors</strong><br>Factorized agent, temporal, and mode attention produces multiple joint trajectory hypotheses.</td>
  </tr>
  <tr>
    <td><strong>🌊 Conditional flow refinement</strong><br>A future-tail-conditioned vector field progressively refines the trajectory priors.</td>
    <td><strong>🌍 Cross-dataset evaluation</strong><br>The manuscript studies INTERACTION reconstruction and zero-shot transfer to 500 Waymo scenes.</td>
  </tr>
</table>

## Contents

- [🎞️ Reconstruction task](#reconstruction-task)
- [🛠️ Getting started](#getting-started)
- [🚀 Training](#training)
- [🔎 Validation](#validation)
- [🚘 Qualitative Gallery](#qualitative-gallery)
- [🙏 Acknowledgements](#acknowledgements)

<a id="reconstruction-task"></a>

## 🎞️ Reconstruction task

<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" srcset="assets/readme/task-window.png">
    <img src="assets/readme/task-window.gif" alt="Task schematic: 10 observed history frames, 20 missing frames to reconstruct, and 5 measured future-tail frames. Both observed sides remain fixed while the missing segment is illustrated in blue." width="100%">
  </picture>
</p>


<a id="getting-started"></a>

## 🛠️ Getting started

### 1. Environment Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/Nova-chen151/TrajBridge.git
cd TrajBridge
conda env create -f environment.yml
conda activate trajbridge
```

### 2. Dataset

Download the [INTERACTION Dataset v1.2](https://interaction-dataset.com/). Here, we only need the data for the multi-agent tracks. After downloading and extracting the zip files, organize the dataset directory as follows:

```
/path/to/INTERACTION_root/
├── maps/
├── test_conditional-multi-agent/
├── test_multi-agent/
├── train/
│   ├── DR_CHN_Merging_ZS0_train
│   ├── ...
└── val/
    ├── DR_CHN_Merging_ZS0_val
    ├── ...

```

Install the map dependency [lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2):
```
pip install lanelet2==1.2.1
```

<a id="training"></a>

## 🚀 Training

```bash
python train.py --root /path/to/INTERACTION_ROOT --train_batch_size 4 --val_batch_size 4 --devices 1
```

<a id="validation"></a>

## 🔎 Validation

```bash
python val.py --root /path/to/INTERACTION_ROOT --ckpt_path runs/<timestamp>/checkpoints/last.ckpt --val_batch_size 4 --devices 1
```
<a id="qualitative-gallery"></a>

## 🚘 Qualitative Gallery

<table>
  <tr>
    <th width="50%">INTERACTION reconstruction</th>
    <th width="50%">Waymo zero-shot transfer</th>
  </tr>
  <tr>
    <td valign="top">
      <picture>
        <source media="(prefers-reduced-motion: reduce)" srcset="assets/readme/interaction-preview.png">
        <img src="assets/readme/interaction-preview.gif" alt="Slideshow of nine static INTERACTION paper examples covering merging, roundabouts, and intersections. Not a temporal rollout." width="100%">
      </picture>
    </td>
    <td valign="top">
      <picture>
        <source media="(prefers-reduced-motion: reduce)" srcset="assets/readme/waymo-preview.png">
        <img src="assets/readme/waymo-preview.gif" alt="Slideshow of nine static Waymo paper examples from the zero-shot transfer study. Not a temporal rollout." width="100%">
      </picture>
    </td>
  </tr>
  <tr>
    <td>Dense multi-agent interactions across merging, roundabouts, and intersections.</td>
    <td>Selected qualitative examples from transfer without fine-tuning.</td>
  </tr>
</table>

<details>
<summary><b>Expand: full-resolution INTERACTION figure and legend</b></summary>

<p align="center">
  <a href="assets/readme/interaction-reconstruction.png">
    <img src="assets/readme/interaction-reconstruction.png" alt="Full nine-panel INTERACTION reconstruction figure with scene names and the original legend." width="100%">
  </a>
</p>

</details>

<details>
<summary><b>Expand: full-resolution Waymo figure and legend</b></summary>

<p align="center">
  <a href="assets/readme/waymo-reconstruction.png">
    <img src="assets/readme/waymo-reconstruction.png" alt="Full nine-panel Waymo zero-shot reconstruction figure with its original vehicle, trajectory, and map legend." width="100%">
  </a>
</p>

</details>

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

We sincerely appreciate [QCNet](https://github.com/ZikangZhou/QCNet) and [HPNet](https://github.com/XiaolongTang23/HPNet) for their awesome codebases.