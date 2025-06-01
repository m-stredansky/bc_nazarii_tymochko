import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def filter_sinter_stones(masks, area_range, min_aspect_ratio, min_iou_with_largest):
    filtered_masks = []
    if not masks:
        return filtered_masks
    
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for i, mask in enumerate(sorted_masks):
        area = mask['area']
        bbox = mask['bbox']
        width = bbox[2]
        height = bbox[3]
        aspect_ratio = min(width / height, height / width)
        
        is_redundant = False
        for larger_mask in sorted_masks[:i]:
            larger_seg = larger_mask['segmentation']
            current_seg = mask['segmentation']
            intersection = np.logical_and(larger_seg, current_seg).sum()
            union = np.logical_or(larger_seg, current_seg).sum()
            iou = intersection / union if union > 0 else 0
            if iou > min_iou_with_largest:
                is_redundant = True
                break
        
        if (area_range[0] <= area <= area_range[1] and 
            aspect_ratio >= min_aspect_ratio and 
            not is_redundant):
            filtered_masks.append(mask)
    
    return filtered_masks

def merge_overlapping_masks(masks, iou_threshold):
    if not masks:
        return masks
    
    merged_masks = []
    remaining_masks = masks.copy()
    
    while remaining_masks:
        current_mask = remaining_masks.pop(0)
        current_seg = current_mask['segmentation']
        current_bbox = current_mask['bbox']
        
        overlapping = []
        for i, other_mask in enumerate(remaining_masks):
            other_seg = other_mask['segmentation']
            intersection = np.logical_and(current_seg, other_seg).sum()
            union = np.logical_or(current_seg, other_seg).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                overlapping.append(i)
        
        for idx in sorted(overlapping, reverse=True):
            other_mask = remaining_masks.pop(idx)
            current_seg = np.logical_or(current_seg, other_mask['segmentation'])
            other_bbox = other_mask['bbox']
            current_bbox = [
                min(current_bbox[0], other_bbox[0]),
                min(current_bbox[1], other_bbox[1]),
                max(current_bbox[0] + current_bbox[2], other_bbox[0] + other_bbox[2]) - min(current_bbox[0], other_bbox[0]),
                max(current_bbox[1] + current_bbox[3], other_bbox[1] + other_bbox[3]) - min(current_bbox[1], other_bbox[1])
            ]
        
        current_mask['segmentation'] = current_seg
        current_mask['bbox'] = current_bbox
        merged_masks.append(current_mask)
    
    return merged_masks

def exclude_text_regions(masks, image_shape, text_regions):
    height, width = image_shape[:2]
    filtered_masks = []
    
    for mask in masks:
        m = mask['segmentation']
        mask_height, mask_width = m.shape
        overlaps_text = False
        
        for x, y, w, h in text_regions:
            mask_x, mask_y = int(x * mask_width / width), int(y * mask_height / height)
            mask_w, mask_h = int(w * mask_width / width), int(h * mask_height / height)
            
            if np.any(m[max(0, mask_y):min(mask_height, mask_y + mask_h),
                        max(0, mask_x):min(mask_width, mask_x + mask_w)]):
                overlaps_text = True
                break
        
        if not overlaps_text:
            filtered_masks.append(mask)
    
    return filtered_masks

def color_segmentation(masks, base_image):
    segmented_image = base_image.copy()
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    
    for i, mask in enumerate(masks):
        m = mask['segmentation']
        m_resized = cv2.resize(m.astype(np.uint8), (segmented_image.shape[1], segmented_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = colors[i % len(colors)]
        color_mask = np.zeros_like(segmented_image, dtype=np.float32)
        
        for c in range(3):
            color_mask[:, :, c] = m_resized * color[c]
        color_mask = np.uint8(color_mask * 0.3 * 255)
        segmented_image = cv2.addWeighted(segmented_image, 1, color_mask, 0.7, 0)
    
    return segmented_image



def get_rotated_min_bounding_rect(mask):
    binary_mask = mask['segmentation'].astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    min_rect = cv2.minAreaRect(largest_contour)
    width, height = min_rect[1]
    
    if width < height:
        width, height = height, width
    
    box_points = cv2.boxPoints(min_rect)
    box_points = box_points.astype(np.int32)
    

    moments = cv2.moments(largest_contour)
    if moments['m00'] == 0: 
        return None
    
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    

    mu20 = moments['mu20'] / moments['m00']  
    mu02 = moments['mu02'] / moments['m00']  
    mu11 = moments['mu11'] / moments['m00']  
    
    
    a = mu20 + mu02
    b = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
    lambda1 = (a + b) / 2  
    lambda2 = (a - b) / 2 
    
    major_axis = 4 * np.sqrt(lambda1)
    minor_axis = 4 * np.sqrt(lambda2)
    
    if mu11 == 0 and mu20 == mu02:
        orientation = 0
    else:
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * 180 / np.pi

    area = (np.pi / 4) * major_axis * minor_axis
    
    equivalent_circle_diameter = np.sqrt(4 * area / np.pi)

    return area, equivalent_circle_diameter, major_axis, minor_axis, orientation, box_points

def draw_bounding_boxes(image, box_points_list, color=(0, 255, 0), thickness=1):
    result_image = image.copy()
    for box_points in box_points_list:
        cv2.polylines(result_image, [box_points], isClosed=True, color=color, thickness=thickness)
    return result_image

def measure_box_dimensions(masks):
    print("\nBounding Box Dimensions (width, height) in pixels:")
    for i, mask in enumerate(masks, 1):
        bbox = mask['bbox']
        width = bbox[2]  
        height = bbox[3]  

 
FOLDER_DIR = Path(__file__).resolve().parents[2]

sam_checkpoint = FOLDER_DIR / 'system_files' / 'SAM'  / 'sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator1_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.65,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  
)

image_original = cv2.imread(FOLDER_DIR / 'system_files' / 'img1.jpg')
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)


image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)  
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  
image = clahe.apply(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)  

WIDTH = 1024
BELT_WIDTH = 1000
HEIGHT = 768

index_of_mm = BELT_WIDTH/image_original.shape[1]
image_resized = cv2.resize(image, (WIDTH, HEIGHT))



masks_original = mask_generator1_.generate(image_resized)
print(f"Number of masks generated: {len(masks_original)}")

text_regions = [
    (0, 0, 310, 50),  # Top-left text
    (image_resized.shape[1] - 63 - 208, image_resized.shape[0] - 72 - (27 // 2), 208, 25)  # Bottom-right text (exact size and position)
]

filtered_masks = exclude_text_regions(masks_original, image_resized.shape, text_regions)


sinter_stone_masks = filter_sinter_stones(filtered_masks, area_range=(10, 5000), min_aspect_ratio=0.6, min_iou_with_largest=0.6)
sinter_stone_masks = merge_overlapping_masks(sinter_stone_masks, iou_threshold=0.6)

equivalent_circle_diameters = []
major_axes = []
minor_axes = []
areas = []
orientations = []
box_points_list = []
for mask in sinter_stone_masks:
    result = get_rotated_min_bounding_rect(mask)
    if result is None:
        continue
    area, eq_diam_px, major_axis_px, minor_axis_px, orientation, box_points = result

    # Convert to mm
    eq_diam_mm = eq_diam_px * index_of_mm
    major_axis_mm = major_axis_px * index_of_mm
    minor_axis_mm = minor_axis_px * index_of_mm
    area_mm2 = area * (index_of_mm ** 2)

    equivalent_circle_diameters.append(eq_diam_mm)
    major_axes.append(major_axis_mm)
    areas.append(area_mm2)
    minor_axes.append(minor_axis_mm)
    orientations.append(orientation)
    box_points_list.append(box_points)

# Print all metrics
print("\nPerimeters, Circumference Diameters, and Legendre Ellipse Properties of Rotated Minimum Bounding Boxes (in pixels):")
for i, (are, diam, major, minor, orient) in enumerate(zip(areas, equivalent_circle_diameters, major_axes, minor_axes, orientations), 1):
    diam_mm = diam * index_of_mm
    major_mm = major * index_of_mm
    minor_mm = minor * index_of_mm
    print(f"Box {i}: Equivalent Diameter = {diam_mm:.2f} mm, Major Axis = {major_mm:.2f} mm, Minor Axis = {minor_mm:.2f} mm, Area = {are * (index_of_mm ** 2):.2f} mm²")



segmented_image = color_segmentation(sinter_stone_masks, image_resized)
segmented_image_with_boxes = draw_bounding_boxes(segmented_image, box_points_list, color=(0, 255, 0), thickness=1)

#SQI Calculation
target_range_min = 5
target_range_max = 50 

num_total_particles = len(equivalent_circle_diameters)
num_target_particles = sum(target_range_min < d < target_range_max for d in equivalent_circle_diameters)

sqi = (num_target_particles / num_total_particles) * 100 if num_total_particles > 0 else 0

print(f"\nSinter Quality Index (SQI): {sqi:.2f}%")

plt.figure(figsize=(50, 25))
plt.imshow(segmented_image_with_boxes)
plt.axis('off')
plt.savefig(FOLDER_DIR / 'output_sam' / 'img1_sam.png')
plt.close()


# counts, bin_edges = np.histogram(equivalent_circle_diameters, bins=40)
# plt.figure(figsize=(12, 6))
# plt.hist(equivalent_circle_diameters, bins=40, color='purple', alpha=0.8, edgecolor='black')
# plt.title('Histogram of Equivalent Circle Diameter')
# plt.xlabel('Equivalent Diameter (mm)')
# plt.ylabel('Frequency')
# plt.xticks(bin_edges, rotation=90, ha='right')
# plt.tight_layout()
# plt.savefig(FOLDER_DIR / 'output_sam' / 'img1_histogram_sam.png')
# plt.close()

counts, bin_edges = np.histogram(equivalent_circle_diameters, bins=40)
plt.figure(figsize=(12, 6))
plt.hist(equivalent_circle_diameters, bins=40, color='purple', alpha=0.8, edgecolor='black')
plt.title(f'Histogram of Equivalent Circle Diameter (SQI = {sqi:.2f}%)')
plt.xlabel('Equivalent Diameter (mm)')
plt.ylabel('Frequency')
plt.xticks(bin_edges, rotation=90, ha='right')
plt.tight_layout()
plt.savefig(FOLDER_DIR / 'output_sam' / 'img1_histogram.png')
plt.close()


combined_mask = np.zeros((image_resized.shape[0], image_resized.shape[1]), dtype=np.uint16)

for i, mask in enumerate(sinter_stone_masks, 1):  # Start labeling from 1
    m = cv2.resize(mask['segmentation'].astype(np.uint8), (image_resized.shape[1], image_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_mask[m == 1] = i  # Assign unique label


def create_mask_from_boxes(box_points_list, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for box_points in box_points_list:
        cv2.fillPoly(mask, [box_points], 255)
    return mask

box_points_list = []
for mask in sinter_stone_masks:
    x, y, w, h = mask['bbox']
    box = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    box_points_list.append(box.astype(np.int32))

bounding_box_mask = create_mask_from_boxes(box_points_list, image_resized.shape)

cv2.imwrite(FOLDER_DIR / 'output_sam'/ 'img1_sam_mask.png', bounding_box_mask)