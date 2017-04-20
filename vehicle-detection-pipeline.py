import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg

from skimage.feature import hog
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# get HOG features
def get_hog_features(rgb_image, colorspace, feature_vector, concatenate_channels):

    if colorspace != 'RGB':
        if colorspace == 'HLS':
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        if colorspace == 'YUV':
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
        elif colorspace == 'Lab':
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
        elif colorspace == 'YCrCb':
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    else:
        image = rgb_image

    image_hog_visualizations = []
    image_hog_features = []

    for channel in range(image.shape[2]):
        if visualize_hog_features:
            channel_hog_features, channel_hog_visualization = hog(image[:,:,channel],
                                          orientations=orient,
                                          pixels_per_cell=(pix_per_cell, pix_per_cell),
                                          cells_per_block=(cell_per_block, cell_per_block),
                                          transform_sqrt=True,
                                          feature_vector=feature_vector,
                                          visualise=visualize_hog_features)
            image_hog_visualizations.append(channel_hog_visualization)
        else:
            channel_hog_features = hog(image[:,:,channel],
                               orientations=orient,
                               pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block),
                               transform_sqrt=True,
                               feature_vector=feature_vector,
                               visualise=visualize_hog_features)

        image_hog_features.append(channel_hog_features)

    if concatenate_channels:
        image_hog_features = np.ravel(image_hog_features)

    if visualize_hog_features:
        return image, image_hog_features, image_hog_visualizations
    else:
        return image, image_hog_features


def get_spatial_features(image):
    image_spatial_features =  cv2.resize(image, (32, 32)).ravel()
    return image_spatial_features


def get_histogram_features(image):
    nbins = 64
    bins_range = (0, 1)
    image_histogram1 = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    image_histogram2 = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    image_histogram3 = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    image_histogram_features = np.concatenate((image_histogram1[0], image_histogram2[0], image_histogram3[0]))
    return image_histogram_features


def extract_features_for_image(rgb_image, colorspace):
    if visualize_hog_features:
        image, image_hog_features, hog_images = get_hog_features(rgb_image, colorspace, feature_vector = True, concatenate_channels = True)
    else:
        image, image_hog_features = get_hog_features(rgb_image, colorspace, feature_vector = True, concatenate_channels = True)

    image_spatial_features = get_spatial_features(image)

    image_histogram_features = get_histogram_features(image)

    return image_hog_features, image_spatial_features, image_histogram_features


def extract_features(image_file_names, colorspace):
    t = time.time()

    features = []
    for image_file_name in image_file_names:
        rgb_image = mpimg.imread(image_file_name)
        image_hog_features, image_spatial_features, image_histogram_features = extract_features_for_image(rgb_image, colorspace)
        image_features = np.hstack((image_spatial_features, image_histogram_features, image_hog_features))
        features.append(image_features)
        # features.append(image_hog_features)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG, Spatial Bins and Histogram features...')
    print (np.shape(features))

    return features


# the training pipeline
def train(colorspace):
    visualize_hog_features = False
    car_images = glob.glob('training/vehicles/*/*.png')
    noncar_images = glob.glob('training/non-vehicles/*/*.png')

    # extract features
    print("Starting feature extraction for car images ..")
    car_features = extract_features(car_images, colorspace)
    print("Starting feature extraction for non-car images ..")
    noncar_features = extract_features(noncar_images, colorspace)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)

    # Fit a per-column scaler
    scaler = StandardScaler().fit(X)
    # Normalize the features by applying the scaler to X
    scaled_X = scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Train a linear SVM classifier
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, scaler


def get_window_list(x_start, x_stop, y_start, y_stop, x_size, y_size, x_overlap, y_overlap):
    # Compute the span of the region to be searched
    x_span = x_stop - x_start
    y_span = y_stop - y_start

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(x_size * (1 - x_overlap))
    ny_pix_per_step = np.int(y_size * (1 - y_overlap))

    # Compute the number of windows in x/y
    nx_buffer = np.int(x_size * x_overlap)
    ny_buffer = np.int(y_size * y_overlap)
    # nx_windows = np.int((x_span - nx_buffer) / nx_pix_per_step)
    # ny_windows = np.int((y_span - ny_buffer) / ny_pix_per_step)
    nx_windows = np.int(x_span / nx_pix_per_step) - 1
    ny_windows = np.int(y_span / ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start
            endx = startx + x_size
            starty = ys * ny_pix_per_step + y_start
            endy = starty + y_size

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# the detection pipeline
def detect_using_sliding_window(rgb_image, colorspace, classifier, scaler, generate_debug_image):
    #image = rgb_image
    image = rgb_image.astype(np.float32)/255

    x_start = 0
    x_stop = image.shape[1]
    y_start = 384
    y_stop = 656
    x_overlap = 0.75
    y_overlap = 0.75

    found_windows = []
    if generate_debug_image:
        debug = np.copy(rgb_image)

    # implement sliding window search
    for size in [96, 122, 128]:
        windows = get_window_list(x_start, x_stop, y_start, y_stop, size, size, x_overlap, y_overlap)
        for window in windows:
            # draw the search area rectangle on the debug image
            if generate_debug_image:
                cv2.rectangle(debug, window[0], window[1], (255, 0, 0), 1)

            # Extract the window patch from original image
            rgb_image_patch = cv2.resize((image[window[0][1]: window[1][1], window[0][0]: window[1][0]]), (64, 64))
            image_patch_hog_features, image_patch_spatial_features, image_patch_histogram_features = \
                extract_features_for_image(rgb_image_patch, colorspace)

            # image_patch_features = np.hstack((image_patch_spatial_features, image_patch_histogram_features, image_patch_hog_features))
            # scaled_features = scaler.transform(image_patch_features.reshape(1, -1))
            scaled_features = np.array(image_patch_hog_features).reshape(1, -1)


            prediction = classifier.predict(scaled_features)
            if prediction == 1:
                found_windows.append(window)
                if generate_debug_image:
                    cv2.rectangle(debug, window[0], window[1], (0, 255, 0), 3)

    # Draw all of the boxes on a mask image
    # mask = np.zeros_like(rgb_image[:,:,0])
    # for bbox in found_windows:
    #     # Draw a filled rectangle given bbox coordinates
    #     cv2.rectangle(mask, bbox[0], bbox[1], (255), -1)

    # Draw all of the boxes on a heatmap image
    heatmap = np.zeros_like(rgb_image[:,:,0])
    for bbox in found_windows:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    heatmap[heatmap <= 2] = 0

    # Find the contours in the mask
    im2, contours, hierarchy = cv2.findContours(heatmap[:,:].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = np.copy(rgb_image)
    for cnt in contours:
        # Get the coordinates of a bounding rect for each contour
        x, y, w, h = cv2.boundingRect(cnt)
        # Draw the bounding rectangles on the result image
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 5)

    if generate_debug_image:
        return result, debug
    else:
        return result


def detect_using_hog_subsampling(rgb_image, colorspace, classifier, scaler, generate_debug_image):
    visualize_hog_features = False

    normalized_rgb_image = rgb_image.astype(np.float32)/255

    y_start = 384
    y_stop = 672
    rgb_image_to_search = normalized_rgb_image[y_start:y_stop, :, :]

    found_windows = []
    if generate_debug_image:
        debug = np.copy(rgb_image)

    for scale in [1.0, 1.5, 2.0]:
        if scale != 1:
            image_shape = rgb_image_to_search.shape
            scaled_rgb_image = cv2.resize(rgb_image_to_search, (np.int(image_shape[1]/scale), np.int(image_shape[0]/scale)))
        else:
            scaled_rgb_image = np.copy(rgb_image_to_search)

        nxblocks = (scaled_rgb_image.shape[1] // pix_per_cell) - 1
        nyblocks = (scaled_rgb_image.shape[0] // pix_per_cell) - 1

        window = 64
        cells_per_step = 2

        nblocks_per_window = (window // pix_per_cell) - 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        scaled_image, scaled_image_hog_features = get_hog_features(scaled_rgb_image, colorspace, feature_vector=False, concatenate_channels = False)
        hog1 = scaled_image_hog_features[0]
        hog2 = scaled_image_hog_features[1]
        hog3 = scaled_image_hog_features[2]

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                scaled_hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                yleft = ypos * pix_per_cell

                subimage = cv2.resize(scaled_image[yleft:yleft + window, xleft:xleft + window], (64, 64))
                scaled_spatial_features = get_spatial_features(subimage)
                scaled_histogram_features = get_histogram_features(subimage)

                scaled_features = scaler.transform(
                    np.hstack((scaled_spatial_features, scaled_histogram_features, scaled_hog_features)).reshape(1, -1))

                prediction = classifier.predict(scaled_features)
                if prediction == 1:
                    xleft_draw = np.int(xleft * scale)
                    yleft_draw = np.int(yleft * scale) + y_start
                    win_draw = np.int(window * scale)
                    found_windows.append(((xleft_draw, yleft_draw), (xleft_draw + win_draw, yleft_draw + win_draw)))
                    if generate_debug_image:
                        cv2.rectangle(debug, (xleft_draw, yleft_draw), (xleft_draw + win_draw, yleft_draw + win_draw), (0, 255, 0), 3)

    # Draw all of the boxes on a mask image
    # mask = np.zeros_like(rgb_image)
    # for bbox in found_windows:
    #     # Draw a filled rectangle given bbox coordinates
    #     cv2.rectangle(mask, bbox[0], bbox[1], (255, 255, 255), -1)

    heatmap = np.zeros_like(rgb_image[:,:,0]).astype(np.uint8)
    for bbox in found_windows:
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    prev_heatmaps.append(heatmap)
    if len(prev_heatmaps) > 16:
        prev_heatmaps.pop(0)

    average_heatmap = np.intc(np.average(prev_heatmaps, 0))

    average_heatmap[average_heatmap <= 2] = 0
    average_heatmap = np.clip(average_heatmap, 0, 255)

    # Find the contours in the mask
    # im2, contours, hierarchy = cv2.findContours(heatmap[ : , : ], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # result = np.copy(rgb_image)
    # for cnt in contours:
    #     # Get the coordinates of a bounding rect for each contour
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     # Draw the bounding rectangles on the result image
    #     cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 5)

    labels = label(average_heatmap)

    result = np.copy(rgb_image)
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(result, bbox[0], bbox[1], (0,0,255), 6)

    if generate_debug_image:
        return result, debug
    else:
        return result


def detect(rgb_image):
    result = detect_using_hog_subsampling(rgb_image, colorspace, classifier, scaler, False)
    # result = detect_using_sliding_window(rgb_image, colorspace, classifier, scaler, False)

    return result


def hog_features_visualization():
    visualize_hog_image = True


def detection_pipeline_exploration(colorspace, classifier, scaler):
    test_images = glob.glob('test_images/*.jpg')
    index = 1
    for image_file_name in test_images:
        print("Testing detection pipeline for " + image_file_name)
        rgb_image = cv2.imread(image_file_name)

        # result, debug = detect_using_sliding_window(rgb_image, colorspace, classifier, scaler, True)
        result, debug = detect_using_hog_subsampling(rgb_image, colorspace, classifier, scaler, True)
        prev_heatmaps.clear()
        cv2.imwrite('debug_'+ str(index) + ".jpg", debug)
        cv2.imwrite('result_'+ str(index) + ".jpg", result)
        cv2.imshow('debug', debug)
        cv2.imshow('result', result)
        index += 1
        cv2.waitKey(0)


# HOG settings for sliding window
# orient = 11
# pix_per_cell = 16
# cell_per_block = 2

# HOG settings for image scaling
orient = 8
pix_per_cell = 8
cell_per_block = 2

prev_heatmaps = []
visualize_hog_features = False


# hog_features_visualization()

# for colorspace in ['RGB', 'YUV', 'HLS', 'Lab', 'YCrCb']:
#     print(colorspace)
#     classifier, scaler = train(colorspace)

colorspace = 'YCrCb'
classifier, scaler = train(colorspace)

# detection_pipeline_exploration(colorspace, classifier, scaler)

# cap = cv2.VideoCapture('test_video.mp4')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         result, debug = detect_using_sliding_window(frame, colorspace, classifier, scaler)
#
#         cv2.imshow('debug',debug)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else: break
#
# cap.release()
# prev_heatmaps.clear()
# cv2.destroyAllWindows()

from moviepy.editor import VideoFileClip
clip_source = VideoFileClip('project_video.mp4')
vid_clip = clip_source.fl_image(detect)
vid_clip.write_videofile('project_video_solution.mp4', audio=False)