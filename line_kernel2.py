'''
Line Kernel2 is for processing a single frame, the 'Process()' function is for import usesself.
This version utilizes intuitive slope scanning
'''

# Essential packages
import cv2
import numpy as np
import argparse

#  Define hyperparameters here
THRESHED_LOW = 33
THRESHED_HIGH = 255
GAUSSIANBLUR_KERNEL = (7,7)
MIN_LINE_VALUE = 150
MIN_LINE_WIDTH = 25
ROI_X_OFFSET = 1/10
ROI_Y_OFFSET = 1/3
INIT_SCAN_Y = 1/10
SHIELD_SIDE_LENGTH = 1/10
SHIELD_INIT_ANGLE = np.pi/2
NUM_SHIELDS = 5


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
    if len(lines_scanned) != 0:     # filter out no lines detected
        for line in lines_scanned:
            line_width, i = line
            midpoint = i - (line_width/2)
            deviation = midpoint - width/2     # distance from mid line, could be positive or negative
            lines_midpoint.append((deviation, line_width))
        lines_midpoint.sort(key=lambda line:abs(line[0]))       # sort from closest to the farthest
        np_lines = np.array(lines_midpoint, np.int32)
        np_lines[:,0] = np_lines[:,0] + int(width/2)

        return np_lines     # (deviation, line_width)
    else:
        return np.array([])

# get the coordinates on a tilted line back as an array
def get_coords_from_line(img,strt_p,end_p):
    # the empty mask to get the coordinate
    mask = np.zeros_like(img)
    cv2.line(mask, strt_p, end_p,(255,255,255),1)       # draw on the mask
    rows, cols = np.where(mask == 255)      # get the lines'coordinates from the mask

    coords = list(tuple(zip(rows,cols)))    # package up to a list of (x,y)

    # sort the coord array with with the distance from the starting point
    strt_p_x, strt_p_y = strt_p
    coords.sort(key = lambda coord:((strt_p_x-coord[0])**2 + (strt_p_y-coord[1])**2), reverse=True)

    return coords

# get the ROI, which is a trapezoid
def get_ROI(img):
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    height, width = get_geometry(img)
    vertices = np.array([[0,height],[ROI_X_OFFSET*width,ROI_Y_OFFSET*height],[(1-ROI_X_OFFSET)*width, ROI_Y_OFFSET*height],[width, height]], np.int32)
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# get the vertix coordinates of the shield from the reference point
# see jupyter notebook shielding.ipynb for further reference
def get_shield_vertex(img, ref_point, theta = SHIELD_INIT_ANGLE):
    _, width = get_geometry(img)
    side = int(SHIELD_SIDE_LENGTH*width)

    x1, y1 = ref_point
    x2 = x1 + side*np.cos(theta)
    y2 = y1 - side*np.sin(theta)

    dx = (side/2)*np.sin(theta)
    dy = (side/2)*np.cos(theta)

    A = (int(x1 - dx), int(y1 - dy))
    B = (int(x2 - dx), int(y2 - dy))
    C = (int(x2 + dx), int(y2 + dy))
    D = (int(x1 + dx), int(y1 + dy))

    return [A,B,C,D]

# perform a single shielding
def propagate(last_p, ref_point, theta):
    pass

# get the midpoint coord of the shield from its four vertices
# img must be single channel
# TODO finish this
def get_midpoint(img, shield_p):
    side_seg_coords = []        # stores the coordinates on the three segments
    for i in range(3):
        strt_p = shield_p[i]
        end_p = shield_p[i+1]
        side_seg = get_coords_from_line(img, strt_p, end_p)
        side_seg_coords.append(side_seg)

    side_seg_values_list = []        # stores the values on the sides of the three segments of shield
    for side_seg in side_seg_coords:
        side_seg_values = []
        for coord in side_seg:
            x, y = coord
            value = img[x,y]
            side_seg_values.append(value)
        side_seg_values_list.append(side_seg_values)

    print(side_seg_values_list)



# contains all transformation in this kernel
def process(input_img):
    # replicate a output img for drawing
    output_img = input_img.copy()

    # get the geometry
    height, width = get_geometry(input_img)

    # cvt to grey than threshold, then get the ROI and blur it
    grey_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2GRAY)
    _,threshed_img = cv2.threshold(grey_img.copy(),THRESHED_LOW,THRESHED_HIGH,cv2.THRESH_BINARY_INV)
    ROI_img = get_ROI(threshed_img.copy())
    blurred_img = cv2.GaussianBlur(ROI_img.copy(), GAUSSIANBLUR_KERNEL, 0)


    # the zeroth scan
    scan_y = int((1-INIT_SCAN_Y)*height)
    init_lines = scan_line(blurred_img, scan_y)
    if init_lines.size != 0:
        # scan for mid point
        center_x = init_lines[0][0]
        print('Line Found')
        cv2.line(output_img, (0,scan_y), (width,scan_y), (255,0,0), 3)
        cv2.circle(output_img, (center_x,scan_y),5,(0,0,255),-1)
        # shield scan
        shield_scan_list = get_shield_vertex(input_img, (center_x, scan_y))
        for i in range(3):
            strt_p = shield_scan_list[i]
            end_p = shield_scan_list[i+1]
            cv2.line(output_img,strt_p,end_p,(0,255,0),3)
        get_midpoint(blurred_img, shield_scan_list)

    # The rest of the scans
    '''
    get slope from last point -> get coord of shield -> get array of shield -> get mid point of shield -> propagate to new point -> repeat
    '''
    for i in range(NUM_SHIELDS):
        pass

    img_bundle = [input_img, output_img, grey_img, threshed_img, ROI_img, blurred_img]
    return img_bundle

def main():
    # get the arguments
    args = get_arguments()
    path = args['image']
    scale = int(args['scale'])

    # read the raw output and resize
    raw_img = cv2.imread(path)
    scaled_img = scale_img(raw_img.copy(), scale)

    # process the img for direction and shift
    output_bundle = process(scaled_img)

    # print every img
    counter = 0
    name = 'input_img,output_img,rey_img,threshold,ROI_img,blurred_img'.split(',')
    for img in output_bundle:
        cv2.imshow(name[counter], img)
        counter += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
