from utils import *
from getImage import readImage
from sklearn.svm import SVC
from getFeature import *
from getFinger import getFinger

np.random.seed(1234)

class getDataset():
    def __init__(self, path, action):
        self.path = path
        self.action = action
    
    def hu_moment(self, path):
        image = readImage(path).threshold_image()
        mhand = getFinger(image).mHand()
        return hu_moment(np.array(mhand))


    def split_data(self):
        label = []
        X = []
        for filename in os.listdir(self.path):
            if filename.split('_')[0] == self.action:
                label.append(1)
            else:
                label.append(0)
            X.append(self.hu_moment(os.path.join(self.path, filename)))
        X = np.array(X)
        #x_train2 = np.array([self.hu_moment(os.path.join(self.path_train_2, filename)) for filename in os.listdir(self.path_train_2)],dtype='uint8')
        # x_test1 = np.array([self.hu_moment(os.path.join(self.path_test_1, filename)) for filename in os.listdir(self.path_test_1)],dtype='uint8')
        # x_test2 = np.array([self.hu_moment(os.path.join(self.path_test_2, filename)) for filename in os.listdir(self.path_test_2)],dtype='uint8')
        
        # create labels
        y = np.array(label)
        # y_train2 = np.ones(x_train2.shape[0])
        # y_test1 = np.zeros(x_test1.shape[0])
        # y_test2 = np.ones(x_test2.shape[0])

        #merge data
        # X_train = np.concatenate((x_train1, x_train2), axis = 0)
        # y_train = np.concatenate((y_train1, y_train2), axis = 0)
        # X_test = np.concatenate((x_test1, x_test2), axis = 0)
        # y_test = np.concatenate((y_test1, y_test2), axis= 0)

        # Shuffle data
        # s = np.arange(X_train.shape[0])
        # np.random.shuffle(s)    
        # X_train = X_train[s]
        # y_train = y_train[s]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True)

        return X_train, X_test, y_train, y_test
    
    def predict(self, x_test):
        model = SVC(kernel = 'linear', C = 1e5)
        model.fit(X_train, y_train)
        return model.predict(x_test)

if __name__ == "__main__":
    data = getDataset("D:/MATERIAL/Image_Processing/dataV1/dataV1/", "G10")
    X_train, X_test, y_train, y_test = data.split_data()
    print('Train data: X={} y={}'.format(X_train.shape, y_train.shape))
    y_predict = data.predict(X_test)
    print(classification_report(y_test, y_predict))


    
    