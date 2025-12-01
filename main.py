"""
PyQt5 GUI for Edge Detection and Object Segmentation
Integrates EdgeDetector, RegionGrowing, and AnnotationPropagator
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QTabWidget, QMessageBox, QSlider,
                             QSpinBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen

# Import the three modules (assumed to be in the same directory)
from modules.preprocessing import EdgeDetector
from modules.segmentation import RegionGrowing
from modules.propagation import AnnotationPropagator

class ImageLabel(QLabel):
    """Custom QLabel for displaying images and capturing mouse clicks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 2px solid black;")
        self.click_callback = None
        self.original_image = None
        self.pixmap_rect = None
        
    def set_click_callback(self, callback):
        self.click_callback = callback
    
    def mousePressEvent(self, event):
        if self.click_callback and self.pixmap() and self.original_image is not None:
            # Get the pixmap's position within the label
            if self.pixmap_rect is None:
                return
            
            # Calculate click position relative to pixmap
            click_x = event.x() - self.pixmap_rect.x()
            click_y = event.y() - self.pixmap_rect.y()
            
            # Check if click is within pixmap bounds
            if (click_x < 0 or click_x >= self.pixmap_rect.width() or
                click_y < 0 or click_y >= self.pixmap_rect.height()):
                return
            
            # Calculate scale factors
            orig_h, orig_w = self.original_image.shape[:2]
            scale_x = orig_w / self.pixmap_rect.width()
            scale_y = orig_h / self.pixmap_rect.height()
            
            # Convert to original image coordinates
            img_x = int(click_x * scale_x)
            img_y = int(click_y * scale_y)
            
            # Clamp to image bounds
            img_x = max(0, min(img_x, orig_w - 1))
            img_y = max(0, min(img_y, orig_h - 1))
            
            print(f"Click at widget ({event.x()}, {event.y()}) -> image ({img_x}, {img_y})")
            self.click_callback(img_x, img_y)
    
    def set_image(self, cv_image):
        """Display OpenCV image in QLabel"""
        if cv_image is None:
            self.clear()
            self.original_image = None
            self.pixmap_rect = None
            return
        
        self.original_image = cv_image
        
        # Convert BGR to RGB
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate where the pixmap will be drawn (centered in label)
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        self.pixmap_rect = scaled_pixmap.rect().translated(x_offset, y_offset)
        
        self.setPixmap(scaled_pixmap)


class EdgeDetectionTab(QWidget):
    """Tab for edge detection functionality"""
    
    def __init__(self):
        super().__init__()
        self.edge_detector = EdgeDetector()
        self.current_image = None
        self.edge_result = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Image display
        self.image_label = ImageLabel()
        layout.addWidget(self.image_label)
        
        # Controls
        control_layout = QHBoxLayout()
        
        # Load image button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Canny", "Sobel", "Iterative Canny"])
        control_layout.addWidget(QLabel("Method:"))
        control_layout.addWidget(self.method_combo)
        
        # Detect edges button
        self.detect_btn = QPushButton("Detect Edges")
        self.detect_btn.clicked.connect(self.detect_edges)
        self.detect_btn.setEnabled(False)
        control_layout.addWidget(self.detect_btn)
        
        # Save result button
        self.save_btn = QPushButton("Save Edge Map")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        layout.addLayout(control_layout)
        
        # Parameters group
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        
        # Gaussian sigma
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 10)
        self.sigma_spin.setValue(int(self.edge_detector.gaussian_sigma))
        param_layout.addRow("Gaussian Sigma:", self.sigma_spin)
        
        # Canny thresholds
        self.low_thresh_spin = QSpinBox()
        self.low_thresh_spin.setRange(1, 255)
        self.low_thresh_spin.setValue(self.edge_detector.canny_low)
        param_layout.addRow("Canny Low:", self.low_thresh_spin)
        
        self.high_thresh_spin = QSpinBox()
        self.high_thresh_spin.setRange(1, 255)
        self.high_thresh_spin.setValue(self.edge_detector.canny_high)
        param_layout.addRow("Canny High:", self.high_thresh_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        self.setLayout(layout)
    
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            self.current_image = self.edge_detector.load_image(filename)
            if self.current_image is not None:
                self.image_label.set_image(self.current_image)
                self.detect_btn.setEnabled(True)
                self.edge_result = None
                self.save_btn.setEnabled(False)
    
    def detect_edges(self):
        if self.current_image is None:
            return
        
        # Update parameters
        self.edge_detector.gaussian_sigma = self.sigma_spin.value()
        self.edge_detector.canny_low = self.low_thresh_spin.value()
        self.edge_detector.canny_high = self.high_thresh_spin.value()
        
        # Get method
        method_map = {
            "Canny": "canny",
            "Sobel": "sobel",
            "Iterative Canny": "iterative"
        }
        method = method_map[self.method_combo.currentText()]
        
        # Detect edges
        self.edge_result = self.edge_detector.get_clean_edges(
            self.current_image, method=method
        )
        
        # Display result
        self.image_label.set_image(self.edge_result)
        self.save_btn.setEnabled(True)
    
    def save_result(self):
        if self.edge_result is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Edge Map", "", "PNG Files (*.png)"
        )
        if filename:
            cv2.imwrite(filename, self.edge_result)
            QMessageBox.information(self, "Success", "Edge map saved successfully!")


class SegmentationTab(QWidget):
    """Tab for object segmentation and annotation propagation"""
    
    def __init__(self):
        super().__init__()
        self.region_grower = RegionGrowing()
        self.propagator = AnnotationPropagator()
        self.current_image = None
        self.original_image = None  # Store the original annotated image
        self.original_mask = None   # Store the original high-quality mask
        self.segmentation_mask = None
        self.display_image = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Image display
        self.image_label = ImageLabel()
        self.image_label.set_click_callback(self.on_image_click)
        layout.addWidget(self.image_label)
        
        # Status label
        self.status_label = QLabel("Load an image to start segmentation")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Controls
        control_layout = QHBoxLayout()
        
        # Load image button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # Undo last seed
        self.undo_btn = QPushButton("Undo Last Click")
        self.undo_btn.clicked.connect(self.undo_last_seed)
        self.undo_btn.setEnabled(False)
        control_layout.addWidget(self.undo_btn)
        
        # Clear all
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setEnabled(False)
        control_layout.addWidget(self.clear_btn)
        
        # Save segmentation
        self.save_seg_btn = QPushButton("Save Segmentation")
        self.save_seg_btn.clicked.connect(self.save_segmentation)
        self.save_seg_btn.setEnabled(False)
        control_layout.addWidget(self.save_seg_btn)
        
        layout.addLayout(control_layout)
        
        # Propagation controls
        prop_layout = QHBoxLayout()
        
        # Load new frame for propagation
        self.load_next_btn = QPushButton("Load Next Frame")
        self.load_next_btn.clicked.connect(self.load_next_frame)
        self.load_next_btn.setEnabled(False)
        prop_layout.addWidget(self.load_next_btn)
        
        # Save propagated result
        self.save_prop_btn = QPushButton("Save Propagated Label")
        self.save_prop_btn.clicked.connect(self.save_propagated)
        self.save_prop_btn.setEnabled(False)
        prop_layout.addWidget(self.save_prop_btn)
        
        layout.addLayout(prop_layout)
        
        # Parameters
        param_group = QGroupBox("Segmentation Parameters")
        param_layout = QFormLayout()
        
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 100)
        self.threshold_spin.setValue(self.region_grower.color_threshold)
        param_layout.addRow("Color Threshold:", self.threshold_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        self.setLayout(layout)
    
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            img = cv2.imread(filename)
            if img is None:
                QMessageBox.warning(self, "Error", f"Couldn't read: {filename}")
                return
            
            # Resize image if larger edge exceeds 500 pixels
            h, w = img.shape[:2]
            max_edge = max(h, w)
            
            if max_edge > 500:
                scale = 500 / max_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Resized from {w}x{h} to {new_w}x{new_h}")
                self.status_label.setText(f"Image resized from {w}x{h} to {new_w}x{new_h}. Click on the object to segment")
            else:
                print(f"Image size {w}x{h} - no resizing needed")
                self.status_label.setText(f"Image size {w}x{h} - Click on the object to segment")
            
            self.current_image = img
            self.display_image = self.current_image.copy()
            self.image_label.set_image(self.display_image)
            self.region_grower.clear_seeds()
            self.segmentation_mask = None
            self.undo_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.save_seg_btn.setEnabled(False)
            self.save_prop_btn.setEnabled(False)
    
    def on_image_click(self, x, y):
        if self.current_image is None:
            return
        
        # Check if coordinates are valid
        h, w = self.current_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            print(f"Click out of bounds: ({x}, {y}), image size: {w}x{h}")
            return
        
        print(f"Valid click at ({x}, {y})")
        
        # Add seed
        self.region_grower.add_seed(x, y)
        
        # Perform segmentation
        threshold = self.threshold_spin.value()
        print(f"Segmenting with threshold: {threshold}")
        mask, contours = self.region_grower.segment(
            self.current_image, threshold=threshold, smooth=True
        )
        
        print(f"Mask sum: {np.sum(mask)}, Contours: {len(contours)}")
        
        # Create visualization
        self.display_image = self.region_grower.create_overlay(
            self.current_image, mask, alpha=0.4, color=(0, 255, 0)
        )
        
        # Draw seed points
        for seed_x, seed_y in self.region_grower.seeds:
            cv2.circle(self.display_image, (seed_x, seed_y), 5, (255, 0, 0), -1)
        
        # Draw contours
        cv2.drawContours(self.display_image, contours, -1, (0, 255, 0), 2)
        
        self.image_label.set_image(self.display_image)
        
        # Enable buttons
        self.undo_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.save_seg_btn.setEnabled(True)
        
        num_seeds = len(self.region_grower.seeds)
        self.status_label.setText(f"Seeds: {num_seeds} | Click to add more seeds")
    
    def undo_last_seed(self):
        if len(self.region_grower.seeds) == 0:
            return
        
        self.region_grower.undo_last_seed()
        
        if len(self.region_grower.seeds) == 0:
            self.display_image = self.current_image.copy()
            self.undo_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.save_seg_btn.setEnabled(False)
            self.status_label.setText("Click on the object to segment")
        else:
            # Re-segment with remaining seeds
            threshold = self.threshold_spin.value()
            mask, contours = self.region_grower.segment(
                self.current_image, threshold=threshold, smooth=True
            )
            
            self.display_image = self.region_grower.create_overlay(
                self.current_image, mask, alpha=0.4, color=(0, 255, 0)
            )
            
            for seed_x, seed_y in self.region_grower.seeds:
                cv2.circle(self.display_image, (seed_x, seed_y), 5, (255, 0, 0), -1)
            
            cv2.drawContours(self.display_image, contours, -1, (0, 255, 0), 2)
            
            num_seeds = len(self.region_grower.seeds)
            self.status_label.setText(f"Seeds: {num_seeds} | Click to add more seeds")
        
        self.image_label.set_image(self.display_image)
    
    def clear_all(self):
        self.region_grower.clear_seeds()
        self.display_image = self.current_image.copy()
        self.image_label.set_image(self.display_image)
        self.undo_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.save_seg_btn.setEnabled(False)
        self.status_label.setText("Click on the object to segment")
    
    def save_segmentation(self):
        if self.region_grower.current_mask is None:
            QMessageBox.warning(self, "Warning", "No segmentation to save!")
            return
        
        # Save mask
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Segmentation Mask", "", "PNG Files (*.png)"
        )
        if filename:
            cv2.imwrite(filename, self.region_grower.current_mask)
            # Store the original image and mask for propagation
            self.original_image = self.current_image.copy()
            self.original_mask = self.region_grower.current_mask.copy()
            self.load_next_btn.setEnabled(True)
            QMessageBox.information(
                self, "Success", 
                "Segmentation saved! Original image and mask stored.\n"
                "You can now load next frames for propagation."
            )
    
    def load_next_frame(self):
        if self.original_mask is None or self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please save a segmentation first!")
            return
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Next Frame", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            next_frame = cv2.imread(filename)
            if next_frame is not None:
                # Propagate annotation using ORIGINAL image and mask
                self.status_label.setText("Propagating annotation from original frame...")
                QApplication.processEvents()
                
                propagated_mask, metrics = self.propagator.propagate_annotation(
                    self.original_image,  # Use original annotated image
                    self.original_mask,   # Use original high-quality mask
                    next_frame
                )
                
                # Update ONLY the current display, not the original
                self.current_image = next_frame
                self.segmentation_mask = propagated_mask
                
                # Create overlay
                self.display_image = self.region_grower.create_overlay(
                    next_frame, propagated_mask, alpha=0.4, color=(0, 255, 0)
                )
                
                # Draw contours
                contours, _ = cv2.findContours(
                    propagated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(self.display_image, contours, -1, (0, 255, 0), 2)
                
                self.image_label.set_image(self.display_image)
                self.save_prop_btn.setEnabled(True)
                
                # Update status
                matches = metrics.get('matches', 0)
                success = metrics.get('transform_success', False)
                self.status_label.setText(
                    f"Propagation from original: {matches} matches | Success: {success}"
                )
    
    def save_propagated(self):
        if self.segmentation_mask is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Propagated Label", "", "PNG Files (*.png)"
        )
        if filename:
            cv2.imwrite(filename, self.segmentation_mask)
            QMessageBox.information(self, "Success", "Propagated label saved!")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation Tool")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Add tabs
        self.edge_tab = EdgeDetectionTab()
        self.seg_tab = SegmentationTab()
        
        tabs.addTab(self.edge_tab, "Edge Detection")
        tabs.addTab(self.seg_tab, "Object Segmentation")
        
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()