#LANE Detection by Hassan Omar Hassan

#steps
# 1---> convert the image to gray ,apply GaussianBlur to reduce noise .
# 2---> apply edge detection using canny algorithm .
# 3---> determine the area of interesting .
# 4---> apply rough transformation , drawing lines on our lane .
# 5---> optimize the drawen lines to be a single line "calculate avr line" .
# 6---> adding our detection to a video

import cv2
import numpy as np
import matplotlib.pyplot as plt



#________________________________________________________________________________
#this method to apply edge detection
def canny_algorithm(image):
    pass
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting the copied image to gray
    blurImage = cv2.GaussianBlur(grayImage ,(5,5) , 0) #apply blur to delete nois
    cannyImage = cv2.Canny(blurImage,50,150) #apply canny algorithm
    return cannyImage
#________________________________________________________________________________




#________________________________________________________________________________
#this method to maske our area of interesting with the edge detected image
# in other words it clears all lines out of the defined triangel "tri"
def region_of_interes(image):
    pass
    height = image.shape[0] #getting the image height
    polygon = np.array( [
    [(200, height), (1100, height), (550, 250)]
    ]) # this is area of interesting
    mask = np.zeros_like(image) # black array "image"
    cv2.fillPoly(mask, polygon, 255) #filling the mask with our area of interesting
    masked_image = cv2.bitwise_and(image ,mask)#bit wise operation to clear anythinf out of our area of interesting
    return masked_image
#________________________________________________________________________________




#________________________________________________________________________________
# this method to draw a blue lines on a black image have same dimensional of lane_image
def dispaly_Lines(image,lines):
    pass
    line_image = np.zeros_like(image) # black array "image"
    # check if lines not empty
    if lines is not None:
        pass
        # loop on lines to add them one by one
        for line in lines:
            pass
            # this step just reshap the line which is defined as an array to 4 variable
            x1,y1,x2,y2 =line.reshape(4)
            #now we can draw the line using 2 points (start , end) , color ,thikness
            cv2.line(line_image ,(x1,y1) , (x2,y2) ,(255,0,0) , 10)
        return line_image
#________________________________________________________________________________




#________________________________________________________________________________
# this method calculate the average of slopes "best fit"
def average_slop_intercept(image ,lines):
    pass
    left_fit=[]
    right_fit=[]
    # loop on lines to calculate the slope one by one
    for line in lines:
        x1,y1,x2,y2 =line.reshape(4)
        #polyfit give us the parameter of each line m,c
        param = np.polyfit( (x1,x2) ,(y1,y2) ,1)
        # getting m
        slope = param[0]
        # getting c
        intercept = param[1]
        if slope > 0 :
            right_fit.append((slope,intercept))
        else:
            left_fit.append((slope,intercept))
    right_fit_avr =np.average(right_fit, axis=0)
    left_fit_avr = np.average(left_fit, axis=0)
    left_line = make_coordinates(image , left_fit_avr)
    right_line = make_coordinates(image , right_fit_avr)
    return np.array([left_line , right_line])
#________________________________________________________________________________




#________________________________________________________________________________
# this mrthod will return array
def make_coordinates(image , line_paramters):
    pass
    # getting m
    m = line_paramters[0]
    # getting c
    c = line_paramters[1]
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-c)/m)
    x2=int((y2-c)/m)
    return np.array([x1,y1,x2,y2])
#________________________________________________________________________________





# finaly drive our algorithm
cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    pass
    # read the current frame
    _,frame = cap.read()
    #calling canny method to apply edge detection
    image_canny = canny_algorithm(frame)
    image_msked = region_of_interes(image_canny)
    # plotting the returned image of canny method
    lines = cv2.HoughLinesP(image_msked , 2 ,np.pi/180 ,100, np.array([]),minLineLength=40 , maxLineGap=5)
    # calling average_slop_intercept to optimize our lines
    average_lines = average_slop_intercept(frame , lines)
    # calling dispaly_Lines to plotting lines on our image
    line_img = dispaly_Lines(frame , average_lines)
    # this to add the lines on the original image but the lines is drawen on the black
    # which is the reurned value from dispaly_Lines so we use addWeighted to put them together
    combo_img = cv2.addWeighted(frame , 0.8 ,line_img ,1,1)
    # viewing the image
    cv2.imshow("result",combo_img)
    # this only # WARNING: one ms betwean the fram and other
    cv2.waitKey(1)
