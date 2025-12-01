# Semi-Automated Image Annotation Tool
**Team Members:** Tuan Minh Dao, Kyle Theodore, Huy Nguyen, Cassio Everling

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python main.py
```
## Running the test
### Test edge detection
```bash
python tests/test_preprocessing.py
```

### Test object segmantation
```bash
python tests/test_segmantation.py input_image.png
```

### Test propagation
```bash
python tests/test_propagation.py
```