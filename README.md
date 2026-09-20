# SAM for Medical Image Segmentation

## Abstract

This project investigates the adaptation of the **Segment Anything Model (SAM)** to medical image segmentation, with a focus on multi-organ abdominal CT images from the **BTCV dataset**. Rather than treating SAM as a fixed natural-image segmentation model, the project develops a progression of increasingly specialized pipelines.

The implementation is organized into three experimental stages:

1. **Zero-shot / prompt-based evaluation** of the pretrained SAM model on medical CT slices.
2. **Task-specific fine-tuning of SAM's mask decoder** using medical segmentation supervision and segmentation-oriented losses.
3. **Joint segmentation and organ classification**, in which a separately trained ResNet-style classifier is integrated with the fine-tuned SAM pipeline.

The codebase therefore moves from evaluating the pretrained foundation model, to domain adaptation, and finally to a segmentation-plus-classification system. The implementation includes support for multiple prompt types, medical-volume preprocessing, class-specific binary masks, Dice/Focal losses, IoU supervision, checkpointing, TensorBoard logging, and 3D mesh visualization.


<img width="1376" height="483" alt="Scientific_progression_diagram_f…_20260920203757" src="https://github.com/user-attachments/assets/ac44c712-33aa-42fc-931a-b7eac4d3750a" />


---



## 1. Research Motivation

Medical image segmentation requires accurate delineation of anatomical structures from imaging modalities such as CT and MRI. Conventional segmentation systems are often trained specifically for a fixed anatomy, dataset, and imaging protocol.

SAM introduced a promptable segmentation paradigm in which a segmentation mask can be generated from visual input together with prompts such as points or bounding boxes. This project explores the following research question:

> **To what extent can a general-purpose promptable segmentation foundation model be adapted to multi-organ medical CT segmentation through prompt engineering, task-specific fine-tuning, and auxiliary anatomical classification?**

The repository implements this investigation experimentally rather than relying only on theoretical discussion.

---

# 2. Research Contributions Implemented in the Code

The project contains several concrete technical contributions.

### 2.1 Medical CT adaptation of SAM

The original SAM pipeline is designed around RGB natural images. This project adapts the input pipeline for grayscale CT slices by:

- loading volumetric medical data using `nibabel`;
- extracting 2D slices from 3D CT volumes;
- converting single-channel CT images into the three-channel representation expected by SAM;
- applying SAM's `ResizeLongestSide` transformation;
- using SAM's image encoder, prompt encoder, and mask decoder for medical segmentation.

This establishes a bridge between a natural-image foundation model and medical CT data.

### 2.2 Prompt-based segmentation experiments

The data pipeline supports multiple SAM prompt strategies:

- **single positive point**
- **multiple points**
- **positive/negative grid points**
- **bounding boxes**
- **mask prompts**

Prompt generation can also be based on:

- anatomical center points;
- random points;
- automatically computed bounding boxes;
- grid-based point selection.

This makes it possible to investigate how different forms of user-like guidance affect segmentation.

### 2.3 Automated anatomical prompt generation

For every anatomical structure present in a CT slice, the code derives a binary mask and computes:

- the anatomical center using a distance transform;
- a bounding box around the structure;
- optional padded bounding boxes;
- positive and negative prompt points.

The center is obtained using the maximum of an Euclidean distance transform, providing a point that tends to lie inside the anatomical structure rather than merely at its geometric centroid.

### 2.4 Task-specific SAM fine-tuning

The second experimental stage fine-tunes SAM for the medical segmentation problem.

The implementation specifically optimizes:

```text
SAM Mask Decoder
```

while the image encoder is evaluated under `torch.no_grad()` during training.

This is a parameter-efficient adaptation strategy: the pretrained visual representation is retained while the mask-generation component is adapted to the medical domain.

### 2.5 Segmentation-specific objective function

The fine-tuning pipeline combines segmentation losses rather than relying on a single generic objective.

The implemented losses include:

- **Focal Loss**
- **Soft Dice Loss**
- **IoU regression loss**

The primary segmentation objective is:

```text
Mask Loss = Focal Loss + λDice × Dice Loss
```

and the total objective is implemented as:

```text
Total Loss =
    Mask Loss
    + λIoU × IoU Loss
```

In the supplied Task 2 and Task 3 configurations, `iou_weight` is set to `0`, meaning the IoU term is implemented but disabled for the configured experiments.

This loss design addresses the strong foreground/background imbalance commonly encountered in medical segmentation.

### 2.6 Joint segmentation and organ classification

The third stage extends the system beyond segmentation.

A custom ResNet-style convolutional classifier is trained to recognize which anatomical class a segmentation mask belongs to. The classifier contains:

- an initial convolution;
- five residual stages;
- batch normalization;
- ReLU activations;
- max pooling;
- dropout;
- a fully connected classification layer.

The classifier is trained separately and then loaded into the joint SAM pipeline.

During joint training, the predicted segmentation mask is passed to the classifier, producing a classification loss:

```text
Classification Loss = CrossEntropy(predicted_organ, ground_truth_organ)
```

The joint model therefore combines:

```text
Segmentation Loss
+
Classification Loss
```

This creates a multi-objective architecture in which the segmentation output is also used as an anatomical representation.

---

# 3. Experimental Pipeline

The repository is divided into three main tasks.

## Task 1 — Pretrained SAM Evaluation

Task 1 evaluates the original pretrained SAM model without fine-tuning it.

The configured experiment uses:

```yaml
model_type: vit_h
prompt_type: single_point
prompt_choice: center
batch_size: 128
```

The workflow is:
<img width="1376" height="768" alt="Medical_CT_segmentation_flowchart_20260920204104" src="https://github.com/user-attachments/assets/9c701749-9f42-4ff7-99f2-2b28d24ec13f" />


For each present anatomical class, the implementation generates a target binary mask and computes the Dice coefficient between the predicted mask and the ground truth.

### Research purpose

Task 1 establishes a **baseline for applying pretrained SAM directly to medical CT images**. It provides a reference point before introducing domain-specific fine-tuning.

---

# 4. Task 2 — Medical Domain Fine-Tuning

Task 2 adapts SAM to the medical segmentation task.

The supplied configuration specifies:

```yaml
model_type: vit_h
prompt_type: single_point
prompt_choice: center
prompt_num: 8
multimask: True
batch_size: 32
max_epoch: 10
optimizer: AdamW
lr: 1.5e-4
```

The model uses the pretrained SAM ViT-H checkpoint and trains the mask decoder.

### Training strategy

The image encoder is used to generate image embeddings under:

```python
with torch.no_grad():
```

The mask decoder remains trainable.

This separates:

- **feature extraction** from
- **medical-domain mask prediction adaptation**.

### Losses

Task 2 implements:

- Focal Loss for pixel-level classification;
- Soft Dice Loss for region overlap;
- optional IoU regression loss.

The Dice coefficient used for evaluation is:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

where:

- \(P\) is the predicted segmentation;
- \(G\) is the ground-truth segmentation.

The validation process calculates Dice scores separately for the anatomical classes and then computes a mean Dice score over classes observed in the validation set.

### Optimization

The implementation supports:

- Adam;
- AdamW;
- linear warm-up;
- cosine learning-rate decay;
- step learning-rate decay.

The supplied Task 2 configuration uses **AdamW**, warm-up, and cosine scheduling.

### Checkpointing

The implementation periodically saves model checkpoints and additionally stores the model corresponding to the best validation mean Dice score.

---

# 5. Task 3 — Joint Segmentation and Organ Classification

Task 3 extends the fine-tuned SAM system into a joint segmentation/classification framework.

The configured experiment uses:

```yaml
model_type: vit_h
prompt_type: bbox
prompt_choice: random
multimask: False
batch_size: 4
max_epoch: 5
optimizer: AdamW
lr: 1e-4
```

The main architectural flow is:
<img width="1376" height="768" alt="Medical_segmentation_and_classif…_20260920204233" src="https://github.com/user-attachments/assets/73f8f703-d2bc-492f-954c-2769884bfbc2" />


The total training objective becomes:

\[
L =
L_{mask}
+
\lambda_{IoU}L_{IoU}
+
L_{classification}
\]

where:

\[
L_{mask}
=
L_{focal}
+
\lambda_{Dice}L_{Dice}
\]

The supplied Task 3 configuration uses:

```yaml
dice_weight: 0.1
iou_weight: 0.0
```

Therefore, for the configured run, the effective objective is primarily:

\[
L =
L_{focal}
+
0.1L_{Dice}
+
L_{classification}
\]

---

# 6. Dataset Processing

The data loader is designed around 3D medical volumes stored in NIfTI format.

The implementation:

1. loads CT volumes using `nibabel`;
2. loads corresponding segmentation labels;
3. iterates through axial slices;
4. removes slices without foreground anatomy for training/validation;
5. extracts each anatomical class independently;
6. converts each class to a binary segmentation mask;
7. computes its center;
8. computes its bounding box;
9. stores the corresponding organ/class identifier.
<img width="1376" height="380" alt="Medical_CT_data_processing_pipeline_20260920204422" src="https://github.com/user-attachments/assets/7a363ac9-f903-4889-a489-138d2c048ba5" />

The system is configured for:

```text
13 foreground anatomical classes
```

with labels represented as integers `1–13`.

Small structures are filtered using a class-dependent pixel-count threshold:

```python
if np.sum(label == i) <= 10 * i:
    continue
```

This prevents extremely small regions from entering the training examples.

---

# 7. Prompt Engineering

Prompt generation is a central component of the project.

## 7.1 Center-point prompts

The anatomical center is determined using:

```text
Euclidean Distance Transform
        ↓
Maximum distance location
        ↓
Interior point of anatomy
```

This is preferable to blindly selecting a corner or arbitrary foreground pixel because the maximum-distance point tends to be safely inside the target structure.

## 7.2 Random positive points

The implementation can randomly sample foreground pixels to simulate less precise user interaction.

## 7.3 Positive/negative multi-point prompts

The system supports combining:

- positive points from inside the organ;
- negative points from the background.

This allows experimentation with richer prompt information.

## 7.4 Bounding-box prompts

For Task 3, automatically generated bounding boxes are used as the SAM prompt.

The bounding box is derived directly from the ground-truth binary mask:

```text
minimum x/y
maximum x/y
        ↓
bounding box
```

Optional padding is supported through the configuration.

## 7.5 Grid-based prompts

A grid-based strategy is also implemented. Points are sampled from a regular image grid and divided into foreground/background prompt candidates according to the ground-truth label.

This provides another experimental mechanism for studying prompt composition.

---

# 8. Model Architecture

## 8.1 Segment Anything Model

The repository includes the SAM implementation under:

```text
segment-anything/
```

The configured backbone is:

```text
ViT-H
```

The relevant components include:

- image encoder;
- prompt encoder;
- mask decoder;
- transformer;
- image preprocessing and resizing;
- predictor interface.

The medical adaptation uses these components as follows:

```text
CT image
   ↓
3-channel conversion
   ↓
SAM preprocessing
   ↓
Image Encoder
   ↓
Image Embedding
   ↓
Prompt Encoder
   ↓
Sparse/Dense Prompt Embeddings
   ↓
Mask Decoder
   ↓
Low-resolution masks
   ↓
SAM post-processing
   ↓
Full-resolution segmentation
```

## 8.2 Auxiliary ResNet classifier

The Task 3 classifier is a custom residual CNN.

Its feature progression is approximately:

```text
1
→ 32
→ 64
→ 128
→ 256
→ 512
→ 1024
```

The network uses residual blocks containing:

- 3×3 convolutions;
- batch normalization;
- ReLU;
- residual shortcuts.

A dropout probability of `0.5` is applied before the final classification stage.

The classifier outputs predictions for the 13 anatomical classes.

---

# 9. Evaluation Methodology

The principal segmentation metric implemented in the repository is the **Dice similarity coefficient**.

For binary masks:

\[
Dice =
\frac{2TP}{2TP + FP + FN}
\]

A score of:

- `1.0` indicates perfect overlap;
- `0.0` indicates no overlap.

The code reports:

- per-organ Dice scores;
- mean Dice across observed organs;
- validation loss;
- mask loss;
- IoU loss;
- classification accuracy in the joint pipeline.

The project also supports TensorBoard logging for training and validation curves.

---

# 10. Visualization and 3D Reconstruction

The repository includes:

```text
task3/visulize3D.py
```

This utility converts a volumetric segmentation result into a 3D polygonal mesh.

The processing pipeline is:

```text
3D segmentation array
        ↓
Extract each anatomical label
        ↓
Marching Cubes
        ↓
Vertices + faces + normals
        ↓
Trimesh object
        ↓
OBJ mesh
```

The implementation uses:

- `skimage.measure.marching_cubes`
- `trimesh`

The resulting mesh can be exported as:

```text
mesh.obj
```

This provides a pathway from 2D slice-wise segmentation to a 3D anatomical representation.

---

# 11. Reproducibility Features

The project includes several mechanisms intended to make experiments reproducible.

### Fixed random seeds

Configurations specify deterministic seeds, for example:

```yaml
seed: 13080
```

The code initializes:

- Python random seed;
- NumPy seed;
- PyTorch seed;
- CUDA seed.

### Configuration-driven experiments

Each task has its own YAML configuration:

```text
task1/cfg.yaml
task2/cfg.yaml
task3/cfg.yaml
```

This separates experimental parameters from implementation code.

### Logging

The training pipelines create:

```text
session.log
params.json
```

and TensorBoard summaries.

### Checkpointing

Training checkpoints are saved periodically, and the fine-tuning pipeline tracks the best validation mean Dice score.

---

# 12. Experimental Progression

The overall project can be interpreted as a three-stage research progression:

| Stage | Model | Prompt | Training | Main Purpose |
|---|---|---|---|---|
| Task 1 | Pretrained SAM ViT-H | Center point | None | Establish direct SAM baseline |
| Task 2 | SAM ViT-H | Center point | Mask decoder fine-tuning | Adapt SAM to medical segmentation |
| Task 3 | SAM ViT-H + ResNet | Bounding box | SAM mask decoder + classifier | Joint segmentation and organ recognition |

This progression is important because it separates three questions:

1. **Can pretrained SAM segment medical anatomy without adaptation?**
2. **Does domain-specific fine-tuning provide a mechanism for adapting SAM to CT anatomy?**
3. **Can anatomical classification be integrated with segmentation to create a richer medical interpretation pipeline?**

---

# 13. What Was Achieved

Based on the supplied source code, the project achieves the following implementation milestones.

### Foundation-model adaptation

A general-purpose SAM implementation was connected to a medical CT dataset and adapted to grayscale volumetric data.

### Promptable medical segmentation

The project implements several automatic prompt-generation strategies, allowing SAM to be evaluated under different forms of segmentation guidance.

### Medical-domain fine-tuning

The SAM mask decoder can be trained using supervised medical segmentation labels rather than relying exclusively on the pretrained natural-image model.

### Segmentation-aware loss design

The implementation combines Focal and Dice losses, explicitly addressing pixel imbalance and region overlap.

### Multi-organ handling

The data pipeline converts multi-class anatomical labels into class-specific binary segmentation problems and evaluates them individually.

### Joint classification and segmentation

The third task introduces a separate residual classifier and connects it to SAM's predicted mask, allowing the system to optimize both segmentation and anatomical classification.

### Quantitative evaluation infrastructure

The code implements per-class Dice evaluation, mean Dice calculation, validation losses, and classification accuracy logging.

### Experiment management

The project provides:

- YAML configurations;
- checkpoint saving;
- session logs;
- parameter snapshots;
- TensorBoard logging;
- reproducible seeds.

### 3D visualization

The segmentation output can be transformed into a 3D mesh, extending the project beyond 2D mask prediction.

---

# 14. Research Interpretation

The project demonstrates a practical pathway for adapting a vision foundation model to a specialized medical imaging problem.

The methodological progression can be summarized as:

```text
Pretrained Foundation Model
          ↓
Prompt-Based Medical Evaluation
          ↓
Medical-Domain Fine-Tuning
          ↓
Segmentation-Specific Optimization
          ↓
Anatomical Classification
          ↓
Joint Medical Segmentation System
          ↓
3D Anatomical Visualization
```

The important research value of the implementation is therefore not limited to a single model checkpoint. It provides an experimental framework for studying how prompt design, parameter-efficient fine-tuning, segmentation losses, and auxiliary anatomical supervision can be combined around SAM.

---

# 16. Software Structure

```text
SAM-for-Medical-Image-Segmentation-mater/
│
├── README.md
├── requirements.txt
│
├── segment-anything/
│   └── Original/extended SAM implementation
│
├── task1/
│   ├── cfg.yaml
│   ├── load_data.py
│   └── main.py
│
├── task2/
│   ├── cfg.yaml
│   ├── load_data.py
│   ├── loss.py
│   ├── main.py
│   ├── model.py
│   └── visualize.py
│
└── task3/
    ├── cfg.yaml
    ├── classifier.py
    ├── dataset_classifier.py
    ├── load_data.py
    ├── loss.py
    ├── main.py
    ├── model.py
    ├── train_classifier.py
    └── visulize3D.py
```

---

# 17. Environment

The supplied `requirements.txt` specifies the principal Python dependencies:

```text
matplotlib==3.7.4
nibabel==5.2.0
numpy==1.24.1
opencv_python==4.8.1.78
PyYAML==6.0.1
scipy==1.11.4
segment_anything==1.0
torch==2.1.1+cu118
tqdm==4.66.1
```

The configurations assume CUDA/GPU execution:

```yaml
use_gpu: True
gpu_idx: 0
```

The SAM ViT-H checkpoint is also required and is referenced by:

```text
sam_vit_h_4b8939.pth
```

---

# 18. Reproduction Outline

A typical reproduction workflow is:

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Install SAM

The project includes a SAM implementation under:

```text
segment-anything/
```

and the original project documentation provides the corresponding installation instructions.

### Step 3 — Prepare BTCV

Download and prepare the BTCV dataset and construct the dataset metadata JSON expected by the loaders.

### Step 4 — Configure paths

Update:

```yaml
dataset_info_path:
data_root_path:
model_root_path:
model_checkpoint:
```

in the appropriate task configuration.

### Step 5 — Run Task 1

```bash
cd task1
python main.py
```

This evaluates the pretrained SAM model with the configured center-point prompt.

### Step 6 — Run Task 2

```bash
cd task2
python main.py
```

This performs SAM mask-decoder fine-tuning.

### Step 7 — Train the Task 3 classifier

```bash
cd task3
python train_classifier.py
```

### Step 8 — Run joint Task 3 training

```bash
cd task3
python main.py
```

### Step 9 — Generate 3D visualization

Use:

```bash
python visulize3D.py
```

with the appropriate segmentation array and output path.

---


# 19. Conclusion

This project implements a complete experimental framework for investigating **SAM-based medical image segmentation on multi-organ CT data**.

The work progresses from direct use of a pretrained foundation model to domain-specific fine-tuning and finally to a joint segmentation/classification architecture. Along the way, it addresses several practical aspects of medical segmentation:

- grayscale-to-SAM input adaptation;
- volumetric CT preprocessing;
- automatic prompt generation;
- class-specific binary masks;
- foreground/background prompt design;
- segmentation-oriented loss functions;
- mask-decoder fine-tuning;
- anatomical classification;
- checkpoint management;
- quantitative Dice evaluation;
- TensorBoard experiment monitoring;
- 3D anatomical reconstruction.

The strongest claim supported by the supplied repository is therefore that a **working experimental pipeline for adapting SAM to multi-organ medical CT segmentation was implemented**, including multiple prompting strategies, supervised fine-tuning, auxiliary classification, evaluation infrastructure, and 3D visualization.

Quantitative claims about performance should be added only after the corresponding experiment logs or result files are available.

---

## References

1. Kirillov, A. et al. **Segment Anything.** International Conference on Computer Vision (ICCV), 2023.
2. Roth, H. R. et al. **DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation.** Relevant BTCV/abdominal CT segmentation benchmark literature.
3. **Beyond the Cranial Vault (BTCV) Multi-Atlas Labeling Challenge** dataset and associated Synapse distribution.
4. The repository includes the Segment Anything implementation used by this project under `segment-anything/`.

---

## Project Status

**Implementation status:** Experimental research pipeline implemented.

**Validated from the supplied source archive:**

- SAM ViT-H integration
- BTCV/NIfTI data loading
- multi-organ slice extraction
- center/bounding-box/multi-point/grid prompting
- SAM baseline evaluation
- mask-decoder fine-tuning
- Focal + Dice optimization
- auxiliary ResNet organ classifier
- joint segmentation/classification training
- checkpointing and logging
- Dice-based validation
- 3D mesh reconstruction

**Not present in the supplied archive:**

- final numerical experiment tables
- complete training logs
- final test-set predictions
- statistically repeated experiments
- independent Task 2/Task 3 test-set evaluation

These items should be added before presenting quantitative performance claims in a formal paper or thesis.
