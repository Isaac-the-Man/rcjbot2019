'''
Line Kernel is for processing a single frame, the 'Process()' function is for import uses
'''
'''
FLOW GRAPH
raw_img -> scaled_img -> grey_img -> threshold -> -> ROI -> blurred ->
'''


# Essential packages
import cv2
import numpy as np
import argparse

#  Define hyperparameters here
THRESHED_LOW = 85
THRESHED_HIGH = 255
ROI_X_OFFSET = 1/10
ROI_Y_OFFSET = 1/3
GAUSSIANBLUR_KERNEL = (7,7)
MIN_LINE_WIDTH = 25     # in terms of pixels
MIN_LINE_VALUE = 150        # 0 ~ 255
SCAN_INIT_Y = 97/100      # portion of height
SCAN_INTERVAL = 1/10      # portion of height
SCAN_COUNT = 6
AVG_SLOPE_COUNT = 3


# get the arguments for frame to frame test
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'enter the image path')
    ap.add_argument('-s', '--scale', required = True, help = 'scale factor of output')
    args = vars(ap.parse_args())        # take in the arguments
    return args

# get the length and width of the image
def get_geometry(img):
    heigth, width = img.shape[:2]
    return heigth, width

# scale the image
def scale_img(img, scale):
    o_h, o_w = get_geometry(img)       # get the geometry of the original image
    s_h = int(o_h/scale)
    s_w = int(o_w/scale)
    scaled_img = cv2.resize(img.copy(), (s_w, s_h))
    print('Original size: %d * %d' %(o_w,o_h))
    print('Scaled size: %d * %d' %(s_w,s_h))
    return scaled_img

# get the ROI, which is a trapezoid
def get_ROI(img):
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    height, width = get_geometry(img)
    vertices = np.array([[0,height],[ROI_X_OFFSET*width,ROI_Y_OFFSET*height],[(1-ROI_X_OFFSET)*width, ROI_Y_OFFSET*height],[width, height]], np.int32)
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# scan a horizontal line on the image and find the center
def scan_line(img, y):
    height, width = get_geometry(img)
    target_line = img[y,0:width]

    # scan any possible lines
    counter = 0
    lines_scanned = []
    for i in range(len(target_line)):
        if target_line[i] >= MIN_LINE_VALUE:
            counter += 1
        else:
            if counter >= MIN_LINE_WIDTH:
                lines_scanned.append([counter, i])
            counter = 0     # reset counter

    # filters out all false lines (find the most middle one)
    lines_midpoint = []     # contains the y coordinate of the line from the most middles to the furthest
    for line in lines_scanned:
        line_width, i = line
        midpoint = i - (line_width/2)
        deviation = midpoint - width/2     # distance from mid line, could be positive or negative
        lines_midpoint.append(deviation)
    lines_midpoint.sort(key=lambda line:abs(line))       # sort from closest to the farthest
    np_lines = np.array(lines_midpoint, np.int32)
    np_lines = np_lines.copy() + int(width/2)

    return np_lines

# scan multiple lines
def scan_lines(img, strt_y, interval_y, epochs):
    scanned_lines = []

    for i in range(epochs):
        lines = scan_line(img, strt_y - i*interval_y)
        if lines.size != 0:
            scanned_lines.append((lines, strt_y - i*interval_y))

    return scanned_lines

# scan for slopes and shifts, returns the instantaneous direction
def scan_direction(img, midpoint_coords):
    shift = 0       # from midline
    direction = 0       # from midline
    height, width = get_geometry(img)

    # if the list is empty (does not found any mid line, go straight foward)
    if not midpoint_coords:
        return shift, direction

    shift = midpoint_coords[0][0] - width/2        # the bottomest midpoint determines the shift

    # scan the slopes
    scanned_slopes = []
    for i in range(len(midpoint_coords)-1):
        x1, y1 = midpoint_coords[i]
        x2, y2 = midpoint_coords[i+1]
        slope = (y1-y2)/(x2-x1)
        scanned_slopes.append(slope)
    np_scanned_slopes = np.array(scanned_slopes)
    print('Slopes found: {}'.format(np_scanned_slopes.size))

    # average the first few slopes
    if np_scanned_slopes.size >= AVG_SLOPE_COUNT:
        avg_slopes = np_scanned_slopes[0:AVG_SLOPE_COUNT]
        direction = np.arctan(np.mean(avg_slopes))
    else:
        direction = np.arctan(np.mean(np_scanned_slopes))

    # correct the direction to start from midline
    direction = ((np.pi/2) - abs(direction))*(1 if direction >= 0 else -1)

    return shift, direction, np_scanned_slopes

# the main pipline
def process(input_img):
    # get the geometry
    height, width = get_geometry(input_img)

    # cvt to grey than threshold, then get the ROI and blur it
    grey_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2GRAY)
    _,threshed_img = cv2.threshold(grey_img.copy(),THRESHED_LOW,THRESHED_HIGH,cv2.THRESH_BINARY_INV)
    ROI_img = get_ROI(threshed_img.copy())
    blurred_img = cv2.GaussianBlur(ROI_img.copy(), GAUSSIANBLUR_KERNEL, 0)

    # scan and render the lines
    scanned_lines = scan_lines(blurred_img, int(SCAN_INIT_Y*height), int(SCAN_INTERVAL*height), SCAN_COUNT)
    midpoint_coords = []
    for lines in scanned_lines:
        lines_y = lines[1]
        lines_x = lines[0][0]
        midpoint_coords.append((lines_x, lines_y))
        print('Midpoints: {},{}'.format(lines_x,lines_y))
        cv2.line(input_img,(0,lines_y),(width,lines_y),(0,0,255),5)
        cv2.circle(input_img,(lines_x,lines_y),5,(255,0,255),-1)

    # scan for slopes and compute direction
    shift, direction, _ = scan_direction(input_img, midpoint_coords)
    print('shift: {}\ndirection: {} deg'.format(shift,direction*180/np.pi))
    for i in range(len(midpoint_coords)-1):
        x1, y1 = midpoint_coords[i]
        x2, y2 = midpoint_coords[i+1]
        print('{} : {} | {} : {}'.format(x1,y1,x2,y2))
        cv2.line(input_img,(x1,y1),(x2,y2),(255,0,0),5)

    # return every img from each step for debug
    img_bundle = [input_img, grey_img, threshed_img, ROI_img, blurred_img]
    return shift, direction, img_bundle


def main():
    # get the arguments
    args = get_arguments()
    path = args['image']
    scale = int(args['scale'])

    # read the raw output and resize
    raw_img = cv2.imread(path)
    scaled_img = scale_img(raw_img.copy(), scale)

    # process the img for direction and shift
    _,_,output_bundle = process(scaled_img)

    # print every img
    name = 'input_img,grey_img,threshed_img,ROI_img,blurred_img'.split(',')
    counter = 0
    for img in output_bundle:
        cv2.imshow(name[counter], img)
        counter += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
