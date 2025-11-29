"""
Main Entry Point for Semi-Automated Annotation Tool
Team: Kyle Theodore, Hy Nguyen, Tuan Minh Dao
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import AnnotationToolGUI


def main():
    """Launch the annotation tool"""
    print("="*60)
    print("Semi-Automated Image Annotation Tool")
    print("="*60)
    print("Team Members:")
    print("  - Kyle Theodore")
    print("  - Hy Nguyen")
    print("  - Tuan Minh Dao")
    print("="*60)
    print("\nStarting application...")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = AnnotationToolGUI()
    window.show()
    
    print("Application ready! Load an image to begin.")
    print("\nQuick Start Guide:")
    print("1. Click 'Load Image' to load a test image")
    print("2. (Optional) Click 'Detect Edges' to see edge detection")
    print("3. Click on the object you want to annotate")
    print("4. Adjust threshold sliders if needed")
    print("5. Add correction clicks if boundary is not perfect")
    print("6. Click 'Save Annotation' to export the mask")
    print("\nFor videos:")
    print("1. Load video with 'Load Video/Frames'")
    print("2. Annotate first frame")
    print("3. Use 'Propagate to Next Frame' to auto-annotate subsequent frames")
    print("="*60)
    
    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()