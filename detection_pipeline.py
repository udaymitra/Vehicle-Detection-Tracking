import numpy as np
from window_search import *
from config import CONFIG
import matplotlib.pyplot as plt
from image_utils import *
from scipy.ndimage.measurements import label
import pickle
from sklearn.externals import joblib
import glob
import matplotlib.image as mpimg

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

def gethotwindows(image, svm, X_scaler, previous=None, count=0):
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500],
                           xy_window=(96, 96), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[425, image.shape[0]],
                            xy_window=(128, 128), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svm, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    # hot_windows = search_windows_fast(image, 400, 520, svm, X_scaler, scale=1.5,
    #                spatial_size=spatial_size, hist_bins=hist_bins,
    #                orient=orient,
    #                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                hist_feat=hist_feat, hog_feat=hog_feat)
    #
    # hot_windows += search_windows_fast(image, 400, image.shape[0], svm, X_scaler, scale=2,
    #            spatial_size=spatial_size, hist_bins=hist_bins,
    #            orient=orient,
    #            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #            hog_channel=hog_channel, spatial_feat=spatial_feat,
    #            hist_feat=hist_feat, hog_feat=hog_feat)

    return hot_windows


def detect_vehicles(image, svm, X_scaler, showheatmap=False, holder=None, debug=False):
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255

    count = 0
    previousLabels = None
    if holder is not None:
        previousLabels = holder.labels
        count = holder.iteration

    hot_windows = gethotwindows(image, svm, X_scaler, previous=previousLabels, count=count)
    if debug:
        window_img_init = draw_boxes(draw_image, hot_windows)
        plt.imshow(window_img_init)
        plt.show()
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap=heat, bbox_list=hot_windows)

    if holder is None:
        holder = ImageHolder()

    if len(holder.previousHeat) < holder.averageCount:
        for i in range(holder.averageCount):
            holder.previousHeat.append(np.copy(heatmap).astype(np.float))

    holder.previousHeat[holder.iteration % holder.averageCount] = heatmap
    total = np.zeros(np.array(holder.previousHeat[0]).shape)

    for value in holder.previousHeat:
        total = total + np.array(value)

    averageHeatMap = total / holder.averageCount

    averageHeatMap = apply_threshold(averageHeatMap, 2)

    if showheatmap:
        plt.imshow(heatmap)
        plt.show()
    labels = label(averageHeatMap)

    holder.labels = labels
    holder.iteration = holder.iteration + 1

    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    window_img = draw_labeled_bboxes(draw_image, labels)
    # print("Showing output")
    return window_img, averageHeatMap


class ImageHolder:
    def __init__(self):
        self.previousHeat = []
        self.labels = []
        self.iteration = 0
        self.averageCount = 10


imageHolder = ImageHolder()

X_scaler = pickle.load(open("scalar.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))

def finalProcessImg(image):
    img,holder = detect_vehicles(image, svm, X_scaler, holder=imageHolder)
    return img

# images = glob.glob("test_images/*")
# for file in [images[0]]:
#     image = mpimg.imread(file)
#     feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#     plt.imshow(feature_image)
#     # plt.show()
#     window_img, heatmap = detect_vehicles(image, svm, X_scaler, debug=False)
#     # plt.imshow(window_img)
#     # plt.show()
#     # plt.imshow(heatmap)
#     # plt.show()


from moviepy.editor import VideoFileClip

output = 'test_video_debug.mp4'
clip = VideoFileClip("test_video.mp4")
output_clip = clip.fl_image(finalProcessImg) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)