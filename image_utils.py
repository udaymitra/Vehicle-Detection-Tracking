import cv2
import numpy as np

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


## CLEAN UP BELOW:
def overlapping(box1,box2):
    return (not ( box1[3] < box2[1] or box1[1] > box2[3] or box1[2] < box2[0] or box1[0] > box2[2] ))


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars

    filterd_boxes = []

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        # bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        if abs(np.min(nonzerox) - np.max(nonzerox)) > 10 and abs(np.min(nonzeroy) - np.max(nonzeroy)) > 10:
            filterd_boxes.append([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])
            # Draw the box on the image

    final_values = []
    for i in filterd_boxes:
        final_value = i
        for j in filterd_boxes:
            if i != j and overlapping(i, j):
                final_value = [min(final_value[0], j[0]),
                               min(final_value[1], j[1]),
                               max(final_value[2], j[2]),
                               max(final_value[3], j[3])]
        cv2.rectangle(img, (final_value[0], final_value[1]), (final_value[2], final_value[3]), (0, 0, 255), 6)
        # Return the image
    return img