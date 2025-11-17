# Computer Vision assignment (17.11.2025)

**Author:** Svyatoslav Suglobov

This repository contains the solution for the Computer Vision assignment dated 17.11.2025.
The main objective is to segment characters and objects from frames of the "Dora the Explorer" animated series.

---

## Models Used for Development

The solution was implemented using:

- **Claude Sonnet 4.5**
- **GPT-5.1-codex**

Initial experiments used **small model variants**. These models are primarily trained on real photographs, which led to expected challenges when working with cartoon images.

---

## Key Scripts

The repository contains several key scripts:

- `detection_segmentation_comparison_improved.py` — improved version using large SAM models (ViT-H 2.4GB, SAM2 Hiera-Large 856MB).
- `cartoon_optimized_detection.py` — settings optimized for cartoon images (ensemble of 6 preprocessed versions).
- `detection_with_large_sam_models.py` — testing large SAM model quality.

Adjusting settings in these files significantly improved detection and segmentation quality:
- **From 1 to 4 objects** in the improved version
- **From 1 to 27 objects** in the ultra-aggressive version

---

## Results

Even in the best configuration (segmenting **27 objects**), the model still made amusing errors:

- Boots → recognized as **teddy bear**
- Mother character → recognized as **cow**
- Sun and balloons → recognized as **sports balls**
- Characters → partially recognized as **apples**, **cakes**, and **oranges**

The results are comical but demonstrate the model's limitations.

---

## Conclusions and Limitations

- Models trained primarily on **real photographs** (COCO dataset) generalize poorly to **drawn/cartoon scenes**.
- For adequate performance on this type of data, one would need:
  - Fine-tuning on specialized datasets with drawings/cartoons;
  - Or using models initially trained on mixed or cartoon data.
- Collecting a dataset and fine-tuning a model within ~3 hours for a homework assignment is practically impossible, so the report shows results "as is".

**However**, through parameter optimization (lowering confidence threshold from 0.3 to 0.05-0.15, using larger models, image preprocessing, Test-Time Augmentation, and ensembling), we achieved a **4-27x improvement** in the number of detected objects.

Nevertheless, this experiment demonstrates characteristic limitations and provides a starting point for further improvements.

---

# Object Detection and Segmentation Comparison

A comprehensive comparison tool for state-of-the-art object detection and segmentation models using COCO 80 classes, optimized for challenging images including cartoon/animated content.

## Overview

This project compares the performance of 7 different computer vision models on object detection and instance segmentation tasks:

### Detection Models
1. **DETR** (DEtection TRansformer) - Facebook's transformer-based detector
2. **YOLO11** - Latest ultralytics YOLO object detector

### Segmentation Models
3. **YOLO11-seg** - YOLO11 with instance segmentation
4. **MaskFormer** - Transformer-based segmentation
5. **Mask2Former** - Improved transformer segmentation
6. **SAM** (Segment Anything Model) - Promptable segmentation with YOLO11 bboxes
7. **SAM2** - Next generation Segment Anything Model with YOLO11 bboxes

## Project Structure

```
.
├── detection_segmentation_comparison.py  # Main script
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── .gitignore                            # Git ignore rules
├── input/                                # Input images directory
│   └── image_to_detect_and_segment.jpg
├── output/                               # Generated output images
│   ├── 1_detr_detection.jpg
│   ├── 2_yolo11_detection.jpg
│   ├── 3_yolo11_segmentation.jpg
│   ├── 4_maskformer_segmentation.jpg
│   ├── 5_mask2former_segmentation.jpg
│   ├── 6_sam_segmentation.jpg
│   └── 7_sam2_segmentation.jpg
├── models/                               # Model checkpoints
│   ├── yolo11n.pt
│   ├── yolo11n-seg.pt
│   ├── sam_vit_b_01ec64.pth
│   └── sam2.1_hiera_base_plus.pt
└── configs/                              # Model configurations
    └── sam2.1_hiera_b+.yaml
```

## Installation

### Basic Requirements

```bash
# Create virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: SAM and SAM2

For SAM support:
```bash
pip install segment-anything
```

For SAM2 support:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Note:** You'll need to download the model checkpoints separately and place them in the `models/` directory.

## Usage

### Basic Usage

```bash
python detection_segmentation_comparison.py
```

### Custom Image

```python
from detection_segmentation_comparison import DetectionSegmentationComparison

comparison = DetectionSegmentationComparison(
    image_path="input/your_image.jpg",
    confidence_threshold=0.3
)
comparison.run_all()
```

### Run Individual Models

```python
# Run only specific models
comparison.run_detr()                  # DETR detection
comparison.run_yolo11_detection()      # YOLO11 detection
comparison.run_yolo11_segmentation()   # YOLO11 segmentation
comparison.run_maskformer()            # MaskFormer segmentation
comparison.run_mask2former()           # Mask2Former segmentation

# SAM requires bounding boxes from a detector
boxes, labels, scores = comparison.run_yolo11_detection()
comparison.run_sam(boxes)              # SAM segmentation
comparison.run_sam2(boxes)             # SAM2 segmentation
```

## Model Resolution Handling

### Models that REQUIRE resizing:
- **DETR**: Resizes to 800x1333 (typical)
- **MaskFormer**: Resizes to fixed input dimensions
- **Mask2Former**: Resizes to fixed input dimensions

### Models that CAN handle arbitrary resolutions:
- **YOLO11**: Flexible, prefers multiples of 32
- **YOLO11-seg**: Same as YOLO11, flexible with sizes
- **SAM**: Natively handles arbitrary resolutions!
- **SAM2**: Natively handles arbitrary resolutions!

## Features

- **COCO 80 Classes**: All models use the standard COCO dataset classes
- **Visual Output**: Generates annotated images with bounding boxes and segmentation masks
- **Confidence Filtering**: Adjustable confidence threshold for all models
- **Flexible Input**: Supports various image resolutions
- **GPU Support**: Automatically uses CUDA if available

## Output

The script generates 5-7 output images (depending on SAM/SAM2 availability):

1. **DETR Detection**: Bounding boxes from DETR
2. **YOLO11 Detection**: Bounding boxes from YOLO11
3. **YOLO11 Segmentation**: Instance masks from YOLO11-seg
4. **MaskFormer Segmentation**: Instance masks from MaskFormer
5. **Mask2Former Segmentation**: Instance masks from Mask2Former
6. **SAM Segmentation**: Masks from SAM using YOLO11 boxes (if installed)
7. **SAM2 Segmentation**: Masks from SAM2 using YOLO11 boxes (if installed)

## Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Ultralytics YOLO
- OpenCV
- Matplotlib
- NumPy

See `requirements.txt` for complete list.

## License

This project is for educational and research purposes.

## Acknowledgments

- DETR: Facebook AI Research
- YOLO11: Ultralytics
- MaskFormer/Mask2Former: Facebook AI Research
- SAM/SAM2: Meta AI Research