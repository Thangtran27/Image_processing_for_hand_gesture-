from utils import *
from getImage import readImage

image_path = './'

def valid_pixel(x, y, img):
    if x >=0 and x < img.shape[0] and y >= 0 and y < img.shape[1]:
      return True
    return False

def extract_each_finger(img, x, y, candSet):
        point = []
        finger = []
        point.append((x, y))

        while(len(point) !=0):
            cx, cy = point.pop()
            if (cx, cy) not in finger:
                finger.append((cx, cy))
            neighbors = [
                (cx - 1, cy),
                (cx - 1, cy - 1),
                (cx - 1, cy + 1),
                (cx, cy -1),
                (cx, cy + 1),
                (cx + 1, cy - 1),
                (cx + 1, cy),
                (cx + 1, cy + 1)
            ]
            for (nx, ny) in neighbors:
                if valid_pixel(nx, ny, img) and (nx, ny) in candSet and (nx, ny) not in finger and (nx, ny) not in point:
                    point.append((nx, ny))
        return finger

class getFinger():
    def __init__(self, image):
        self.image = image
        self.mhand = self.mHand()
        self.mcontour = self.mContour()
        self.M0, self.R0 = self.center_R0()
        self.canset = self.candSet()
        

    def mHand(self):
        mHand = []
        for cx in range(self.image.shape[0]):
            for cy in range(self.image.shape[1]):
                neighbors = [
                (cx - 1, cy),
                (cx - 1, cy - 1),
                (cx - 1, cy + 1),
                (cx, cy -1),
                (cx, cy + 1),
                (cx + 1, cy - 1),
                (cx + 1, cy),
                (cx + 1, cy + 1)
                ]
                is_interior = True
                for (nx, ny) in neighbors:
                    if valid_pixel(nx, ny, self.image) and self.image[nx, ny] == 0:
                        is_interior = False
                        break
                if(is_interior):
                    mHand.append((cx, cy))
        return mHand
    
    def mContour(self):
        mContour = []
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.image[i, j] == 255 and (i, j) not in self.mhand:
                    mContour.append((i, j))
        return mContour

    def center_R0(self):
        R0 = 0
        M0 = []
        for i in self.mhand:
            distances = pairwise.euclidean_distances([i], Y=self.mcontour)[0]
            min_distance = distances[distances.argmin()]
            if min_distance >=R0:
                R0 = min_distance
                M0 = i
        return M0, R0
    
    def candSet(self):
        candSet = []
        # calculate R_average
        distance_palm = pairwise.euclidean_distances([self.M0], Y=self.mcontour)[0]
        distance_sum = distance_palm.sum()
        R_avg = distance_sum/len(distance_palm)
        #print(R_avg)
        for i in self.mcontour:
            distances = pairwise.euclidean_distances([i], Y=[self.M0])
            if distances >= int(R_avg)+1:
                candSet.append(i)
        return candSet
    

    def get_mFinger(self):
        mFinger = []
        mFingertip = []
        while(len(self.canset) != 0):
            x, y = self.canset[0]
            each_finger= []
            each_finger = extract_each_finger(self.image, x, y, self.canset)
            distan = pairwise.euclidean_distances([self.M0], Y=each_finger)[0]
            point_max = each_finger[distan.argmax()]
            for (cx, cy) in each_finger:
                self.canset.remove((cx, cy))
            each_finger.remove(point_max)
            mFingertip.append(point_max) # finger tip
            mFinger.append([point_max, each_finger])      
        return mFingertip, mFinger # mFingers = [[fingertip, [finger]], [...[...]], ...]

if __name__ == '__main__':
    image = readImage("test.jpg").threshold_image()
    hand = getFinger(image).mHand()
    print(type(hand))
    # mFingertip, mFinger = hand.get_mFinger()
    # print(mFingertip)
    # print(len(mFinger))



