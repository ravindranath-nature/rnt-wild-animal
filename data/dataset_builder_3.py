import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Configuration
base_path = r'D:\dev\research\Animal_Detection\raw\raw_v4'
classes = classes =   [
    "boar",
    "elephant",
    "wolf",
    "Antelope",
    "fox",
    "gorilla",
    "koala",
    "pocupine",
    "Bison"
  ]

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_yolo_label(label_path):
    """Read YOLO format label file"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines if line.strip()]

def write_yolo_label(label_path, annotations):
    """Write YOLO format label file"""
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(' '.join(map(str, ann)) + '\n')

def update_class_ids(base_path, classes):
    """Task 1: Update class IDs in labels according to class index"""
    print("Task 1: Updating class IDs...")
    
    for class_idx, class_name in enumerate(classes):
        labels_path = os.path.join(base_path, class_name, 'original', 'labels')
        
        if not os.path.exists(labels_path):
            print(f"Warning: Labels path not found for {class_name}")
            continue
            
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_path, label_file)
                annotations = read_yolo_label(label_path)
                
                # Update class IDs
                updated_annotations = []
                for ann in annotations:
                    if len(ann) >= 5:  # YOLO format: class_id x_center y_center width height
                        ann[0] = str(class_idx)  # Update class ID
                        updated_annotations.append(ann)
                
                write_yolo_label(label_path, updated_annotations)
    
    print("Task 1 completed: Class IDs updated")

def augment_contrast(image, factor):
    """Apply contrast augmentation to image"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def augment_images_and_labels(base_path, classes):
    """Task 2: Augment images and labels with contrast variations"""
    print("Task 2: Augmenting images and labels...")
    
    # All augmentations for classes with < 900 images
    all_contrast_factors = [1.05, 0.95, 1.03, 0.97, 1.01, 0.99]  # +5%, -5%, +3%, -3%, +1%, -1%
    all_contrast_names = ['contrast_p5', 'contrast_m5', 'contrast_p3', 'contrast_m3', 'contrast_p1', 'contrast_m1']
    
    # Limited augmentations for classes with 900-1200+ images
    limited_contrast_factors = [1.05, 0.95, 1.03, 0.97]  # +5%, -5%, +3%, -3%
    limited_contrast_names = ['contrast_p5', 'contrast_m5', 'contrast_p3', 'contrast_m3']
    
    for class_name in classes:
        print(f"Processing class: {class_name}")
        
        original_images_path = os.path.join(base_path, class_name, 'original', 'images')
        original_labels_path = os.path.join(base_path, class_name, 'original', 'labels')
        
        augmented_images_path = os.path.join(base_path, class_name, 'augmented', 'images')
        augmented_labels_path = os.path.join(base_path, class_name, 'augmented', 'labels')
        
        ensure_dir(augmented_images_path)
        ensure_dir(augmented_labels_path)
        
        if not os.path.exists(original_images_path):
            print(f"Warning: Original images path not found for {class_name}")
            continue
        
        # Count images to determine augmentation strategy
        image_files = [f for f in os.listdir(original_images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_count = len(image_files)
        
        # Select augmentation strategy based on image count
        if image_count >= 900:
            contrast_factors = limited_contrast_factors
            contrast_names = limited_contrast_names
            print(f"  {class_name}: {image_count} images - Using limited augmentations (+5%, -5%, +3%, -3%)")
        else:
            contrast_factors = all_contrast_factors
            contrast_names = all_contrast_names
            print(f"  {class_name}: {image_count} images - Using all augmentations (+5%, -5%, +3%, -3%, +1%, -1%)")
        
        # Copy original images and labels to augmented folder
        for img_file in image_files:
            # Copy original image
            src_img = os.path.join(original_images_path, img_file)
            dst_img = os.path.join(augmented_images_path, f"original_{img_file}")
            shutil.copy2(src_img, dst_img)
                
            # Copy corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(original_labels_path, label_file)
            dst_label = os.path.join(augmented_labels_path, f"original_{label_file}")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            
            # Load original image for augmentation
            image = cv2.imread(src_img)
            if image is None:
                continue
            
            # Apply contrast augmentations
            for factor, name in zip(contrast_factors, contrast_names):
                # Augment image
                aug_image = augment_contrast(image, factor)
                aug_img_name = f"{name}_{img_file}"
                aug_img_path = os.path.join(augmented_images_path, aug_img_name)
                cv2.imwrite(aug_img_path, aug_image)
                
                # Copy label (labels don't change with contrast)
                if os.path.exists(src_label):
                    aug_label_name = f"{name}_{label_file}"
                    aug_label_path = os.path.join(augmented_labels_path, aug_label_name)
                    shutil.copy2(src_label, aug_label_path)
    
    print("Task 2 completed: Image and label augmentation finished")

def split_dataset(base_path, classes, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """Task 3: Split augmented data into train/valid/test"""
    print("Task 3: Splitting dataset...")
    
    for class_name in classes:
        print(f"Splitting class: {class_name}")
        
        augmented_images_path = os.path.join(base_path, class_name, 'augmented', 'images')
        augmented_labels_path = os.path.join(base_path, class_name, 'augmented', 'labels')
        
        split_images_path = os.path.join(base_path, class_name, 'split', 'images')
        split_labels_path = os.path.join(base_path, class_name, 'split', 'labels')
        
        # Create split directories
        for split in ['train', 'valid', 'test']:
            ensure_dir(os.path.join(split_images_path, split))
            ensure_dir(os.path.join(split_labels_path, split))
        
        if not os.path.exists(augmented_images_path):
            print(f"Warning: Augmented images path not found for {class_name}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(augmented_images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle for random splitting
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        valid_count = int(total_files * valid_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        valid_files = image_files[train_count:train_count + valid_count]
        test_files = image_files[train_count + valid_count:]
        
        splits = {
            'train': train_files,
            'valid': valid_files,
            'test': test_files
        }
        
        # Copy files to respective splits
        for split_name, files in splits.items():
            for img_file in files:
                # Copy image
                src_img = os.path.join(augmented_images_path, img_file)
                dst_img = os.path.join(split_images_path, split_name, img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy corresponding label
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(augmented_labels_path, label_file)
                dst_label = os.path.join(split_labels_path, split_name, label_file)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
        
        print(f"  {class_name}: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
    
    print("Task 3 completed: Dataset splitting finished")

def merge_splits(base_path, classes):
    """Task 4: Merge all class splits into final dataset structure"""
    print("Task 4: Merging splits...")
    
    final_dataset_path = os.path.join(base_path, 'datasetv4')
    
    # Create final dataset structure
    for split in ['train', 'valid', 'test']:
        ensure_dir(os.path.join(final_dataset_path, split, 'images'))
        ensure_dir(os.path.join(final_dataset_path, split, 'labels'))
    
    for class_name in classes:
        print(f"Merging class: {class_name}")
        
        class_split_path = os.path.join(base_path, class_name, 'split')
        
        if not os.path.exists(class_split_path):
            print(f"Warning: Split path not found for {class_name}")
            continue
        
        for split in ['train', 'valid', 'test']:
            src_images = os.path.join(class_split_path, 'images', split)
            src_labels = os.path.join(class_split_path, 'labels', split)
            
            dst_images = os.path.join(final_dataset_path, split, 'images')
            dst_labels = os.path.join(final_dataset_path, split, 'labels')
            
            # Copy images
            if os.path.exists(src_images):
                for img_file in os.listdir(src_images):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # Add class prefix to avoid naming conflicts
                        new_name = f"{class_name}_{img_file}"
                        src_path = os.path.join(src_images, img_file)
                        dst_path = os.path.join(dst_images, new_name)
                        shutil.copy2(src_path, dst_path)
            
            # Copy labels
            if os.path.exists(src_labels):
                for label_file in os.listdir(src_labels):
                    if label_file.endswith('.txt'):
                        # Add class prefix to avoid naming conflicts
                        new_name = f"{class_name}_{label_file}"
                        src_path = os.path.join(src_labels, label_file)
                        dst_path = os.path.join(dst_labels, new_name)
                        shutil.copy2(src_path, dst_path)
    
    print("Task 4 completed: Dataset merging finished")

def create_dataset_yaml(base_path, classes):
    """Create dataset.yaml file for YOLO training"""
    yaml_content = f"""# Dataset configuration for YOLO
path: {os.path.join(base_path, 'datasetv4')}
train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(base_path, 'datasetv4', 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created at: {yaml_path}")

def get_image_dimensions(image_path):
    """Get image dimensions without loading the full image"""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except:
        return None

def generate_dataset_summary(base_path, classes):
    """Generate comprehensive dataset summary"""
    print("\n" + "="*70)
    print("COMPREHENSIVE DATASET SUMMARY")
    print("="*70)
    
    summary_data = {}
    total_original = 0
    total_augmented = 0
    image_dimensions = set()
    
    # Collect statistics for each class
    for class_name in classes:
        class_data = {
            'original_count': 0,
            'augmented_count': 0,
            'dimensions': set()
        }
        
        # Count original images
        original_path = os.path.join(base_path, class_name, 'original', 'images')
        if os.path.exists(original_path):
            original_files = [f for f in os.listdir(original_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            class_data['original_count'] = len(original_files)
            
            # Sample a few images to get dimensions
            for i, img_file in enumerate(original_files[:5]):  # Check first 5 images
                img_path = os.path.join(original_path, img_file)
                dims = get_image_dimensions(img_path)
                if dims:
                    class_data['dimensions'].add(dims)
                    image_dimensions.add(dims)
        
        # Count augmented images
        augmented_path = os.path.join(base_path, class_name, 'augmented', 'images')
        if os.path.exists(augmented_path):
            augmented_files = [f for f in os.listdir(augmented_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            class_data['augmented_count'] = len(augmented_files)
        
        summary_data[class_name] = class_data
        total_original += class_data['original_count']
        total_augmented += class_data['augmented_count']
    
    # Create summary text for both console and file output
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("COMPREHENSIVE DATASET SUMMARY")
    summary_lines.append("="*70)
    summary_lines.append(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Class-wise summary
    summary_lines.append(f"{'CLASS':<15} {'ORIGINAL':<10} {'AUGMENTED':<12} {'DIMENSIONS':<20}")
    summary_lines.append("-" * 70)
    
    for class_name in classes:
        data = summary_data[class_name]
        dims_str = ", ".join([f"{w}x{h}" for w, h in sorted(data['dimensions'])])
        if len(dims_str) > 18:
            dims_str = dims_str[:15] + "..."
        
        line = f"{class_name:<15} {data['original_count']:<10} {data['augmented_count']:<12} {dims_str:<20}"
        summary_lines.append(line)
        print(line)
    
    summary_lines.append("-" * 70)
    total_line = f"{'TOTAL':<15} {total_original:<10} {total_augmented:<12}"
    summary_lines.append(total_line)
    print(total_line)
    
    # Dataset split summary
    summary_lines.append("")
    summary_lines.append("="*50)
    summary_lines.append("FINAL DATASET SPLITS")
    summary_lines.append("="*50)
    
    final_path = os.path.join(base_path, 'datasetv4')
    split_totals = {}
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(final_path, split, 'images')
        if os.path.exists(split_path):
            count = len([f for f in os.listdir(split_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            split_totals[split] = count
            line = f"{split.upper():<8}: {count:>6} images"
            summary_lines.append(line)
            print(line)
    
    total_final = sum(split_totals.values())
    total_line = f"{'TOTAL':<8}: {total_final:>6} images"
    summary_lines.append(total_line)
    print(total_line)
    
    # Image dimensions summary
    summary_lines.append("")
    summary_lines.append("="*50)
    summary_lines.append("IMAGE DIMENSIONS FOUND")
    summary_lines.append("="*50)
    
    if image_dimensions:
        sorted_dims = sorted(image_dimensions, key=lambda x: x[0] * x[1])
        for width, height in sorted_dims:
            pixels = width * height
            line = f"{width} x {height} pixels ({pixels:,} total pixels)"
            summary_lines.append(line)
            print(line)
    else:
        line = "No image dimensions could be determined"
        summary_lines.append(line)
        print(line)
    
    # Augmentation strategy summary
    summary_lines.append("")
    summary_lines.append("="*50)
    summary_lines.append("AUGMENTATION STRATEGY APPLIED")
    summary_lines.append("="*50)
    
    for class_name in classes:
        original_count = summary_data[class_name]['original_count']
        if original_count >= 900:
            strategy = "Limited (±5%, ±3%)"
            multiplier = 5  # original + 4 augmentations
        else:
            strategy = "Full (±5%, ±3%, ±1%)"
            multiplier = 7  # original + 6 augmentations
        
        expected = original_count * multiplier
        actual = summary_data[class_name]['augmented_count']
        
        line = f"{class_name:<15}: {strategy:<20} (Expected: {expected}, Actual: {actual})"
        summary_lines.append(line)
        print(line)
    
    # Add processing details
    summary_lines.append("")
    summary_lines.append("="*50)
    summary_lines.append("PROCESSING DETAILS")
    summary_lines.append("="*50)
    summary_lines.append(f"Base Path: {base_path}")
    summary_lines.append(f"Classes: {', '.join(classes)}")
    summary_lines.append(f"Number of Classes: {len(classes)}")
    summary_lines.append(f"Split Ratios: Train 80%, Valid 10%, Test 10%")
    summary_lines.append("")
    summary_lines.append("Augmentation Types Applied:")
    summary_lines.append("- Contrast +5% and -5%")
    summary_lines.append("- Contrast +3% and -3%")
    summary_lines.append("- Contrast +1% and -1% (only for classes < 900 images)")
    
    summary_lines.append("")
    summary_lines.append("="*70)
    summary_lines.append("SUMMARY COMPLETE")
    summary_lines.append("="*70)
    
    # Save summary to file
    summary_file_path = os.path.join(base_path, 'dataset_summary.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nSummary saved to: {summary_file_path}")
    except Exception as e:
        print(f"Warning: Could not save summary file: {str(e)}")
    
    # Also save as CSV for easy analysis
    csv_file_path = os.path.join(base_path, 'dataset_summary.csv')
    try:
        import csv
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write class statistics
            writer.writerow(['Class Statistics'])
            writer.writerow(['Class', 'Original Count', 'Augmented Count', 'Dimensions', 'Strategy'])
            
            for class_name in classes:
                data = summary_data[class_name]
                dims_str = ", ".join([f"{w}x{h}" for w, h in sorted(data['dimensions'])])
                original_count = data['original_count']
                strategy = "Limited (±5%, ±3%)" if original_count >= 900 else "Full (±5%, ±3%, ±1%)"
                
                writer.writerow([
                    class_name, 
                    data['original_count'], 
                    data['augmented_count'], 
                    dims_str,
                    strategy
                ])
            
            writer.writerow([])  # Empty row
            
            # Write split statistics
            writer.writerow(['Split Statistics'])
            writer.writerow(['Split', 'Count'])
            for split, count in split_totals.items():
                writer.writerow([split.capitalize(), count])
            writer.writerow(['Total', sum(split_totals.values())])
            
            writer.writerow([])  # Empty row
            
            # Write dimensions
            writer.writerow(['Image Dimensions'])
            writer.writerow(['Width', 'Height', 'Total Pixels'])
            if image_dimensions:
                for width, height in sorted(image_dimensions, key=lambda x: x[0] * x[1]):
                    writer.writerow([width, height, width * height])
        
        print(f"CSV summary saved to: {csv_file_path}")
    except Exception as e:
        print(f"Warning: Could not save CSV summary: {str(e)}")
    
    print(f"\nSummary files saved in: {base_path}")
    return summary_data

def main():
    """Main execution function"""
    print("Starting dataset processing pipeline...")
    print(f"Base path: {base_path}")
    print(f"Classes: {classes}")
    print("-" * 50)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    try:
        # Task 1: Update class IDs
        update_class_ids(base_path, classes)
        
        # Task 2: Augment images and labels
        augment_images_and_labels(base_path, classes)
        
        # Task 3: Split dataset
        split_dataset(base_path, classes)
        
        # Task 4: Merge splits
        merge_splits(base_path, classes)
        
        # Create dataset YAML file
        create_dataset_yaml(base_path, classes)
        
        # Print statistics
        generate_dataset_summary(base_path, classes)
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final dataset available at: {os.path.join(base_path, 'datasetv4')}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Pipeline execution failed!")

if __name__ == "__main__":
    main()