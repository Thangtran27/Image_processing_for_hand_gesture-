from utils import *
from getFinger import getFinger
from getImage import readImage
from getFeature import *




if __name__ =='__main__':
    image_1 = readImage('G10_002628.jpg').threshold_image() #G06_000055.jpg
    image_2 = readImage('G07_003272.jpg').threshold_image() #G07_003272.jpg
    hand_1 = getFinger(image_1).mHand()
    hand_2 = getFinger(image_2).mHand()
    hu_1 = hu_moment(np.array(hand_1))
    hu_2 = hu_moment(np.array(hand_2))
    print(hu_1)
    print(hu_2)
    print(distance(hu_1, hu_2))


    