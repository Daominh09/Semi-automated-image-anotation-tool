import cv2
import numpy as np
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.segmentation import RegionGrowing

def _interactive_demo(path):
    img = cv2.imread(path)
    if img is None:
        print("couldn't read:", path)
        return
    rg = RegionGrowing()
    win = "RegionGrowing - Segmentation"
    cv2.namedWindow(win)
    disp = img.copy()
    
    def update():
        nonlocal disp
        if rg.seeds:
            m,_ = rg.segment(img)
            d = rg.create_overlay(img,m)
        else:
            d = img.copy()
        for sx,sy in rg.seeds:
            cv2.circle(d,(sx,sy),3,(0,0,255),-1)
        disp = d
        cv2.imshow(win,disp)
    
    def mouse_cb(ev,x,y,flags,param):
        if ev==cv2.EVENT_LBUTTONDOWN:
            rg.add_seed(x,y)
            print("added:",(x,y))
            update()
    
    cv2.setMouseCallback(win,mouse_cb)
    print("demo running... keys: u=undo, c=clear, s=save, q=quit")
    update()
    
    while True:
        k = cv2.waitKey(10)&0xFF
        if k==ord('q'): break
        elif k==ord('u'):
            rg.undo_last_seed(); print("undo"); update()
        elif k==ord('c'):
            rg.clear_seeds(); print("clear"); update()
        elif k==ord('s'):
            if rg.current_mask is not None:
                os.makedirs(os.path.join(PROJECT_ROOT, 'test_images', 'test_object_segmantation'), exist_ok=True)
                cv2.imwrite(os.path.join(PROJECT_ROOT, 'test_images', 'test_object_segmantation', 'test_mask.png'), rg.current_mask)
                break
            else:
                print("no mask to save - add seeds first")
    cv2.destroyAllWindows()


if __name__=="__main__":
    import sys
    print("RegionGrowing loaded")
    print("defaults:",RegionGrowing.DEFAULT_COLOR_THRESHOLD,RegionGrowing.DEFAULT_GRAY_THRESHOLD,RegionGrowing.DEFAULT_MORPH_KERNEL)
    if len(sys.argv)<2:
        print("usage: python region_growing.py img.jpg")
    else:
        _interactive_demo(sys.argv[1])