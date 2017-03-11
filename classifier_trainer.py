import glob
import numpy as np
import time
import pickle
from config import CONFIG

from featurizer import extract_features
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

color_space = CONFIG['color_space']
orient = CONFIG['orient']
pix_per_cell = CONFIG['pix_per_cell']
cell_per_block = CONFIG['cell_per_block']
hog_channel = CONFIG['hog_channel']
spatial_size = CONFIG['spatial_size']
hist_bins = CONFIG['hist_bins']
spatial_feat = CONFIG['spatial_feat']
hist_feat = CONFIG['hist_feat']
hog_feat = CONFIG['hog_feat']

def train_svm():
    carFiles = glob.glob("data/vehicles/*/*.png", recursive=True)
    print("Loading car files")
    cars = []
    for image in carFiles:
        cars.append(image)
    print("Loaded car files")
    nonCarFiles = glob.glob("data/non-vehicles/*/*.png", recursive=False)
    notcars = []
    print("Loading non-car files")
    for image in nonCarFiles:
        notcars.append(image)

    print("Extracting car features")
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    print("Extracting non-car features")
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # filename = "svm.pkl"
    # joblib.dump(svc, filename)
    pickle.dump(svc, open("svm.pkl", "wb"))
    pickle.dump(X_scaler, open("scalar.pkl","wb"))

def main():
    train_svm()

if __name__ == "__main__":
    main()