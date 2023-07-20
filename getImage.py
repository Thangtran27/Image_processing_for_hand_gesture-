import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

def get_folder(path):
    return os.listdir(path)

class readImage():
    def __init__(self, path):
        self.path = path
    
    def imread(self):
        return cv2.imread(self.path)
    
    def threshold_image(self):
        image = self.imread()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold

if __name__ == '__main__':
    image = readImage("test.jpg")
    cv2.imshow('test', image.threshold_image())
    cv2.waitKey(0)
    cv2.destroyWindows()
