# Semi-Automated Image Annotation Tool
**Team Members:** Kyle Theodore, Hy Nguyen, Tuan Minh Dao

## Project Structure
```
annotation_tool/
│
├── main.py                 # Entry point (Person C)
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py   # Person A - Edge detection
│   ├── segmentation.py    # Person B - Region growing
│   ├── propagation.py     # Person D - Feature matching
│   └── metrics.py         # All - Evaluation metrics
│
├── gui/
│   ├── __init__.py
│   └── main_window.py     # Person C - GUI implementation
│
├── data/
│   ├── input/            # Test images/videos
│   ├── ground_truth/     # Manual annotations
│   └── output/           # Results
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_propagation.py
│   └── test_metrics.py
│
└── docs/
    ├── parameters.md      # Recommended parameters
    └── results.xlsx       # Accuracy measurements

```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python main.py
```

## Module Dependencies
- **Person C (GUI)** needs: preprocessing.py, segmentation.py, propagation.py
- **Person B (Segmentation)** needs: preprocessing.py (for edge maps)
- **Person D (Propagation)** needs: segmentation.py (for refinement)
- **All members** need: metrics.py (for evaluation)

## Development Workflow
1. Each person develops their module independently
2. Test with provided test scripts
3. Person C integrates all modules into GUI
4. Final testing with complete dataset

## Accuracy Metrics
- **IoU (Intersection over Union)**: overlap / union
- **Dice Coefficient**: 2 * overlap / (area1 + area2)
- **Target**: IoU > 0.7, Dice > 0.8