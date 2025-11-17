"""
ULTRA-AGGRESSIVE Cartoon/Anime Optimized Detection Script
Specifically tuned for stylized, animated, and cartoon images
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class CartoonOptimizedDetection:
    """Ultra-aggressive detection optimized for cartoon/anime images"""

    def __init__(self, image_path, confidence_threshold=0.05):
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold

        # Load image
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]

        print(f"Original image size: {self.width}x{self.height}")

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create multiple preprocessed versions
        self.preprocessed_images = self.create_multiple_preprocessed_versions()

    def create_multiple_preprocessed_versions(self):
        """Create multiple preprocessed versions for ensemble detection"""
        print("\nCreating multiple preprocessed versions...")

        versions = {}
        pil_image = Image.fromarray(self.original_image_rgb)

        # Version 1: High contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        img_high_contrast = enhancer.enhance(1.5)
        versions['high_contrast'] = np.array(img_high_contrast)

        # Version 2: Enhanced edges
        img_edges = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        enhancer = ImageEnhance.Sharpness(img_edges)
        img_edges = enhancer.enhance(1.5)
        versions['enhanced_edges'] = np.array(img_edges)

        # Version 3: Saturated colors
        enhancer = ImageEnhance.Color(pil_image)
        img_saturated = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Contrast(img_saturated)
        img_saturated = enhancer.enhance(1.3)
        versions['saturated'] = np.array(img_saturated)

        # Version 4: Brightness adjusted
        enhancer = ImageEnhance.Brightness(pil_image)
        img_bright = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(img_bright)
        img_bright = enhancer.enhance(1.2)
        versions['bright'] = np.array(img_bright)

        # Version 5: Original with bilateral filter
        img_bilateral = cv2.bilateralFilter(self.original_image_rgb, 9, 75, 75)
        versions['bilateral'] = img_bilateral

        print(f"Created {len(versions)} preprocessed versions")
        return versions

    def merge_detections(self, all_detections, iou_threshold=0.5):
        """Merge detections from multiple runs using weighted NMS"""
        if not all_detections:
            return [], [], []

        # Combine all boxes, scores, labels
        all_boxes = []
        all_scores = []
        all_labels = []

        for detection in all_detections:
            boxes, labels, scores = detection
            all_boxes.extend(boxes)
            all_labels.extend(labels)
            all_scores.extend(scores)

        if not all_boxes:
            return [], [], []

        all_boxes = np.array(all_boxes)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Apply soft NMS - keep more overlapping detections
        keep_indices = []

        for class_id in np.unique(all_labels):
            class_mask = all_labels == class_id
            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]
            class_indices = np.where(class_mask)[0]

            # Sort by score
            sorted_indices = np.argsort(class_scores)[::-1]

            for i in sorted_indices:
                if class_scores[i] < 0.01:  # Skip very low confidence
                    continue

                # Check IoU with already kept boxes
                keep = True
                box_i = class_boxes[i]

                for j in keep_indices:
                    if all_labels[j] != class_id:
                        continue

                    box_j = all_boxes[j]
                    iou = self.calculate_iou(box_i, box_j)

                    if iou > iou_threshold:
                        # If boxes overlap significantly, keep the one with higher score
                        if all_scores[class_indices[i]] <= all_scores[j]:
                            keep = False
                            break

                if keep:
                    keep_indices.append(class_indices[i])

        return all_boxes[keep_indices], all_labels[keep_indices], all_scores[keep_indices]

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def run_yolo_ensemble(self):
        """Run YOLO on multiple image versions and merge results"""
        print("\n" + "="*60)
        print("ULTRA-AGGRESSIVE CARTOON DETECTION")
        print("Running ensemble detection on multiple image versions...")
        print("="*60)

        # Use largest YOLO model
        print("\nLoading YOLO11x model (may download ~140MB if not cached)...")
        model = YOLO('yolo11x.pt')

        all_detections = []

        # Run on original
        print("\n[1/6] Detecting on original image...")
        results = model(
            self.image_path,
            conf=self.confidence_threshold,
            iou=0.3,  # Lower for more overlapping detections
            augment=True,
            imgsz=1280,
            verbose=False
        )[0]

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)
            all_detections.append((boxes, labels, scores))
            print(f"  → Found {len(boxes)} detections")

        # Run on preprocessed versions
        for idx, (name, preprocessed_img) in enumerate(self.preprocessed_images.items(), 2):
            print(f"\n[{idx}/6] Detecting on {name} version...")

            # Save temporary file
            temp_path = f"temp_{name}.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR))

            results = model(
                temp_path,
                conf=self.confidence_threshold,
                iou=0.3,
                augment=True,
                imgsz=1280,
                verbose=False
            )[0]

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)
                all_detections.append((boxes, labels, scores))
                print(f"  → Found {len(boxes)} detections")

            # Clean up
            Path(temp_path).unlink()

        # Merge all detections
        print("\n" + "="*60)
        print("Merging detections from all versions...")
        boxes, labels, scores = self.merge_detections(all_detections, iou_threshold=0.5)

        print(f"\n✓ FINAL RESULT: {len(boxes)} unique objects detected")
        print("="*60)

        return boxes, labels, scores

    def draw_results(self, boxes, labels, scores):
        """Visualize detection results"""
        fig, ax = plt.subplots(1, figsize=(14, 10))
        ax.imshow(self.original_image_rgb)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(COCO_CLASSES)))

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            color = colors[label % len(colors)]
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            ax.text(x1, y1-5, f'{class_name}: {score:.2f}',
                   bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

        ax.set_title(f"Ultra-Aggressive Cartoon Detection\nTotal Detections: {len(boxes)}",
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        output_path = 'output/ultra_aggressive_cartoon_detection.jpg'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved results to: {output_path}")
        plt.close()

        # Print detection summary
        print("\n" + "="*60)
        print("DETECTION SUMMARY:")
        print("="*60)

        class_counts = {}
        for label, score in zip(labels, scores):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(score)

        for class_name, scores_list in sorted(class_counts.items()):
            avg_conf = np.mean(scores_list)
            print(f"  {class_name}: {len(scores_list)} detection(s) (avg conf: {avg_conf:.2f})")

        print("="*60)


def main():
    print("="*70)
    print("ULTRA-AGGRESSIVE CARTOON/ANIME DETECTION SCRIPT")
    print("="*70)
    print("\nOptimizations:")
    print("  ✓ Ultra-low confidence threshold (0.05)")
    print("  ✓ Ensemble detection on 6 image versions")
    print("  ✓ Multiple preprocessing strategies")
    print("  ✓ Soft NMS for overlapping detections")
    print("  ✓ YOLO11x (largest, most accurate model)")
    print("  ✓ Test-Time Augmentation")
    print("  ✓ 1280px input resolution")
    print("="*70)

    detector = CartoonOptimizedDetection(
        image_path="input/image_to_detect_and_segment.jpg",
        confidence_threshold=0.05  # Very aggressive!
    )

    boxes, labels, scores = detector.run_yolo_ensemble()
    detector.draw_results(boxes, labels, scores)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()