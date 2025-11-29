"""
Person C: GUI & Interaction Logic
Main application window with all controls and display
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSlider, QFileDialog, 
                            QGroupBox, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import os
import time
import json

# Import project modules
import sys
sys.path.append('..')
from modules.preprocessing import EdgeDetector
from modules.segmentation import RegionGrowing
from modules.propagation import AnnotationPropagator
from modules.metrics import AnnotationMetrics, ResultsLogger


class ImageLabel(QLabel):
    """Custom QLabel that handles mouse clicks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
    
    def mousePressEvent(self, event):
        """Handle mouse click events"""
        if event.button() == Qt.LeftButton and self.parent_window:
            # Convert click coordinates to image coordinates
            x = int((event.x() - self.offset.x()) / self.scale_factor)
            y = int((event.y() - self.offset.y()) / self.scale_factor)
            
            # Send to parent window
            self.parent_window.handle_image_click(x, y)


class AnnotationToolGUI(QMainWindow):
    """Main GUI window for the annotation tool"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize modules
        self.edge_detector = EdgeDetector()
        self.region_grower = RegionGrowing()
        self.propagator = AnnotationPropagator(feature_type='ORB')
        self.results_logger = ResultsLogger()
        
        # State variables
        self.current_image = None
        self.current_mask = None
        self.current_contours = []
        self.edge_map = None
        self.video_frames = []
        self.current_frame_idx = 0
        self.frame_masks = {}  # Store masks for each frame
        
        # Setup UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Semi-Automated Annotation Tool')
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel: Display area
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, stretch=3)
        
        # Status bar
        self.statusBar().showMessage('Ready. Load an image to begin.')
    
    def create_control_panel(self):
        """Create the control panel with all buttons and sliders"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # File loading group
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton('Load Image')
        self.load_image_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_image_btn)
        
        self.load_video_btn = QPushButton('Load Video/Frames')
        self.load_video_btn.clicked.connect(self.load_video)
        file_layout.addWidget(self.load_video_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Edge detection group
        edge_group = QGroupBox("Edge Detection (Person A)")
        edge_layout = QVBoxLayout()
        
        # Gaussian sigma slider
        edge_layout.addWidget(QLabel('Gaussian Sigma:'))
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(5)
        self.sigma_slider.setMaximum(50)
        self.sigma_slider.setValue(15)
        self.sigma_slider.valueChanged.connect(self.update_edge_detection)
        edge_layout.addWidget(self.sigma_slider)
        self.sigma_label = QLabel('1.5')
        edge_layout.addWidget(self.sigma_label)
        
        # Canny low threshold
        edge_layout.addWidget(QLabel('Canny Low Threshold:'))
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setMinimum(10)
        self.canny_low_slider.setMaximum(200)
        self.canny_low_slider.setValue(50)
        self.canny_low_slider.valueChanged.connect(self.update_edge_detection)
        edge_layout.addWidget(self.canny_low_slider)
        self.canny_low_label = QLabel('50')
        edge_layout.addWidget(self.canny_low_label)
        
        # Canny high threshold
        edge_layout.addWidget(QLabel('Canny High Threshold:'))
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setMinimum(50)
        self.canny_high_slider.setMaximum(300)
        self.canny_high_slider.setValue(150)
        self.canny_high_slider.valueChanged.connect(self.update_edge_detection)
        edge_layout.addWidget(self.canny_high_slider)
        self.canny_high_label = QLabel('150')
        edge_layout.addWidget(self.canny_high_label)
        
        self.detect_edges_btn = QPushButton('Detect Edges')
        self.detect_edges_btn.clicked.connect(self.detect_edges)
        edge_layout.addWidget(self.detect_edges_btn)
        
        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)
        
        # Segmentation group
        seg_group = QGroupBox("Segmentation (Person B)")
        seg_layout = QVBoxLayout()
        
        # Region growing threshold
        seg_layout.addWidget(QLabel('Region Growing Threshold (τ):'))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(5)
        self.threshold_slider.setMaximum(50)
        self.threshold_slider.setValue(20)
        seg_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel('20')
        seg_layout.addWidget(self.threshold_label)
        
        seg_layout.addWidget(QLabel('Click on image to segment'))
        
        self.add_click_btn = QPushButton('Add Correction Click')
        self.add_click_btn.setCheckable(True)
        seg_layout.addWidget(self.add_click_btn)
        
        self.undo_btn = QPushButton('Undo Last Click')
        self.undo_btn.clicked.connect(self.undo_click)
        seg_layout.addWidget(self.undo_btn)
        
        self.clear_btn = QPushButton('Clear All Clicks')
        self.clear_btn.clicked.connect(self.clear_clicks)
        seg_layout.addWidget(self.clear_btn)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        # Video navigation group
        nav_group = QGroupBox("Video Navigation")
        nav_layout = QVBoxLayout()
        
        self.frame_label = QLabel('Frame: 0/0')
        nav_layout.addWidget(self.frame_label)
        
        btn_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton('← Previous')
        self.prev_frame_btn.clicked.connect(self.previous_frame)
        self.prev_frame_btn.setEnabled(False)
        btn_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton('Next →')
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        btn_layout.addWidget(self.next_frame_btn)
        nav_layout.addLayout(btn_layout)
        
        self.propagate_btn = QPushButton('Propagate to Next Frame')
        self.propagate_btn.clicked.connect(self.propagate_annotation)
        self.propagate_btn.setEnabled(False)
        nav_layout.addWidget(self.propagate_btn)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Save group
        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout()
        
        self.save_annotation_btn = QPushButton('Save Annotation')
        self.save_annotation_btn.clicked.connect(self.save_annotation)
        save_layout.addWidget(self.save_annotation_btn)
        
        self.save_results_btn = QPushButton('Save Results (Excel)')
        self.save_results_btn.clicked.connect(self.save_results)
        save_layout.addWidget(self.save_results_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        layout.addStretch()
        
        return panel
    
    def create_display_panel(self):
        """Create the display panel for images"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Image display
        self.image_display = ImageLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("border: 2px solid black;")
        self.image_display.setMinimumSize(800, 600)
        layout.addWidget(self.image_display)
        
        # Display options
        options_layout = QHBoxLayout()
        
        self.show_original_btn = QPushButton('Original')
        self.show_original_btn.setCheckable(True)
        self.show_original_btn.setChecked(True)
        self.show_original_btn.clicked.connect(self.update_display)
        options_layout.addWidget(self.show_original_btn)
        
        self.show_edges_btn = QPushButton('Edges')
        self.show_edges_btn.setCheckable(True)
        self.show_edges_btn.clicked.connect(self.update_display)
        options_layout.addWidget(self.show_edges_btn)
        
        self.show_mask_btn = QPushButton('Mask')
        self.show_mask_btn.setCheckable(True)
        self.show_mask_btn.clicked.connect(self.update_display)
        options_layout.addWidget(self.show_mask_btn)
        
        self.show_contours_btn = QPushButton('Contours')
        self.show_contours_btn.setCheckable(True)
        self.show_contours_btn.setChecked(True)
        self.show_contours_btn.clicked.connect(self.update_display)
        options_layout.addWidget(self.show_contours_btn)
        
        layout.addLayout(options_layout)
        
        return panel
    
    def load_image(self):
        """Load a single image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Image', '', 
            'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if filepath:
            self.current_image = cv2.imread(filepath)
            if self.current_image is not None:
                self.video_frames = []
                self.current_frame_idx = 0
                self.frame_masks = {}
                self.region_grower.clear_seeds()
                self.current_mask = None
                self.update_display()
                self.statusBar().showMessage(f'Loaded: {os.path.basename(filepath)}')
            else:
                QMessageBox.warning(self, 'Error', 'Could not load image')
    
    def load_video(self):
        """Load video or multiple frames"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Video', '', 
            'Videos (*.mp4 *.avi *.mov);;All Files (*.*)'
        )
        
        if filepath:
            cap = cv2.VideoCapture(filepath)
            self.video_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.video_frames.append(frame)
            
            cap.release()
            
            if self.video_frames:
                self.current_frame_idx = 0
                self.current_image = self.video_frames[0]
                self.frame_masks = {}
                self.region_grower.clear_seeds()
                self.update_display()
                self.update_frame_navigation()
                self.statusBar().showMessage(
                    f'Loaded {len(self.video_frames)} frames'
                )
            else:
                QMessageBox.warning(self, 'Error', 'Could not load video')
    
    def update_edge_detection(self):
        """Update edge detection parameters from sliders"""
        sigma = self.sigma_slider.value() / 10.0
        self.sigma_label.setText(f'{sigma:.1f}')
        self.edge_detector.gaussian_sigma = sigma
        
        canny_low = self.canny_low_slider.value()
        self.canny_low_label.setText(str(canny_low))
        self.edge_detector.canny_low = canny_low
        
        canny_high = self.canny_high_slider.value()
        self.canny_high_label.setText(str(canny_high))
        self.edge_detector.canny_high = canny_high
    
    def detect_edges(self):
        """Detect edges in current image"""
        if self.current_image is None:
            QMessageBox.warning(self, 'Error', 'No image loaded')
            return
        
        self.edge_map = self.edge_detector.get_clean_edges(
            self.current_image, 
            method='iterative',
            cleanup_method='morphological'
        )
        
        self.update_display()
        self.statusBar().showMessage('Edge detection complete')
    
    def handle_image_click(self, x, y):
        """Handle click on image for segmentation"""
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return
        
        # Add seed
        self.region_grower.add_seed(x, y)
        
        # Update threshold
        threshold = self.threshold_slider.value()
        self.threshold_label.setText(str(threshold))
        
        # Perform segmentation
        self.current_mask, self.current_contours = self.region_grower.segment(
            self.current_image, 
            threshold=threshold
        )
        
        # Store mask for current frame
        if self.video_frames:
            self.frame_masks[self.current_frame_idx] = self.current_mask.copy()
        
        self.update_display()
        self.statusBar().showMessage(f'Added seed at ({x}, {y})')
    
    def undo_click(self):
        """Undo last click"""
        self.region_grower.undo_last_seed()
        
        if len(self.region_grower.seeds) == 0:
            self.current_mask = None
            self.current_contours = []
        else:
            threshold = self.threshold_slider.value()
            self.current_mask, self.current_contours = self.region_grower.segment(
                self.current_image, 
                threshold=threshold
            )
        
        self.update_display()
        self.statusBar().showMessage('Undid last click')
    
    def clear_clicks(self):
        """Clear all clicks"""
        self.region_grower.clear_seeds()
        self.current_mask = None
        self.current_contours = []
        self.update_display()
        self.statusBar().showMessage('Cleared all clicks')
    
    def update_display(self):
        """Update the image display"""
        if self.current_image is None:
            return
        
        # Start with original or edge map
        if self.show_edges_btn.isChecked() and self.edge_map is not None:
            display_img = cv2.cvtColor(self.edge_map, cv2.COLOR_GRAY2BGR)
        else:
            display_img = self.current_image.copy()
        
        # Add mask overlay
        if self.show_mask_btn.isChecked() and self.current_mask is not None:
            display_img = self.region_grower.create_overlay(
                display_img, 
                self.current_mask,
                alpha=0.3
            )
        
        # Draw contours
        if self.show_contours_btn.isChecked() and self.current_contours:
            cv2.drawContours(display_img, self.current_contours, -1, 
                           (0, 255, 0), 2)
        
        # Convert to QPixmap and display
        height, width = display_img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(display_img.data, width, height, bytes_per_line, 
                      QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit display
        scaled_pixmap = pixmap.scaled(self.image_display.size(), 
                                     Qt.KeepAspectRatio, 
                                     Qt.SmoothTransformation)
        
        self.image_display.setPixmap(scaled_pixmap)
        
        # Update scale factor for click coordinates
        self.image_display.scale_factor = scaled_pixmap.width() / width
    
    def update_frame_navigation(self):
        """Update frame navigation buttons and labels"""
        if self.video_frames:
            self.frame_label.setText(
                f'Frame: {self.current_frame_idx + 1}/{len(self.video_frames)}'
            )
            self.prev_frame_btn.setEnabled(self.current_frame_idx > 0)
            self.next_frame_btn.setEnabled(
                self.current_frame_idx < len(self.video_frames) - 1
            )
            self.propagate_btn.setEnabled(
                self.current_mask is not None and 
                self.current_frame_idx < len(self.video_frames) - 1
            )
        else:
            self.frame_label.setText('Frame: 0/0')
            self.prev_frame_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)
            self.propagate_btn.setEnabled(False)
    
    def previous_frame(self):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.current_image = self.video_frames[self.current_frame_idx]
            
            # Load mask if exists
            if self.current_frame_idx in self.frame_masks:
                self.current_mask = self.frame_masks[self.current_frame_idx]
                contours, _ = cv2.findContours(self.current_mask, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                self.current_contours = list(contours)
            else:
                self.current_mask = None
                self.current_contours = []
            
            self.region_grower.clear_seeds()
            self.update_display()
            self.update_frame_navigation()
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame_idx < len(self.video_frames) - 1:
            self.current_frame_idx += 1
            self.current_image = self.video_frames[self.current_frame_idx]
            
            # Load mask if exists
            if self.current_frame_idx in self.frame_masks:
                self.current_mask = self.frame_masks[self.current_frame_idx]
                contours, _ = cv2.findContours(self.current_mask, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                self.current_contours = list(contours)
            else:
                self.current_mask = None
                self.current_contours = []
            
            self.region_grower.clear_seeds()
            self.update_display()
            self.update_frame_navigation()
    
    def propagate_annotation(self):
        """Propagate annotation to next frame (Person D)"""
        if self.current_mask is None or not self.video_frames:
            return
        
        if self.current_frame_idx >= len(self.video_frames) - 1:
            return
        
        # Get current and next frame
        prev_frame = self.video_frames[self.current_frame_idx]
        next_frame = self.video_frames[self.current_frame_idx + 1]
        
        # Propagate
        self.statusBar().showMessage('Propagating annotation...')
        propagated_mask, metrics = self.propagator.propagate_annotation(
            prev_frame, 
            self.current_mask,
            next_frame,
            refine=True,
            region_grower=self.region_grower
        )
        
        # Move to next frame and show result
        self.current_frame_idx += 1
        self.current_image = next_frame
        self.current_mask = propagated_mask
        self.frame_masks[self.current_frame_idx] = propagated_mask
        
        # Extract contours
        contours, _ = cv2.findContours(propagated_mask, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        self.current_contours = list(contours)
        
        self.update_display()
        self.update_frame_navigation()
        self.statusBar().showMessage(
            f'Propagated! Matches: {metrics["matches"]}'
        )
    
    def save_annotation(self):
        """Save current annotation"""
        if self.current_mask is None:
            QMessageBox.warning(self, 'Error', 'No annotation to save')
            return
        
        # Save mask as PNG
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Mask', '', 'PNG Image (*.png)'
        )
        
        if save_path:
            cv2.imwrite(save_path, self.current_mask)
            
            # Also save contours as JSON
            if self.current_contours:
                contour_path = save_path.replace('.png', '_contours.json')
                contours_data = []
                for contour in self.current_contours:
                    contours_data.append(contour.squeeze().tolist())
                
                with open(contour_path, 'w') as f:
                    json.dump(contours_data, f)
            
            self.statusBar().showMessage(f'Saved annotation to {save_path}')
    
    def save_results(self):
        """Save evaluation results"""
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Results', '', 'Excel File (*.xlsx)'
        )
        
        if save_path:
            self.results_logger.save_to_excel(save_path)
            QMessageBox.information(self, 'Success', 'Results saved!')


# Run the application
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = AnnotationToolGUI()
    window.show()
    sys.exit(app.exec_())