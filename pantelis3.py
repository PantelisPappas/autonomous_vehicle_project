import cv2
import numpy as np
from motor import kinhsh, motor_init

def extrapolate_lines(lines, image, color=[255, 0, 0], thickness = 10):


##  xone = []
##  yone = []
##  xtwo = []
##  ytwo = []
  # segregate the small line segments into the left lane group or right lane group
  if lines is not None:  
    for line in lines:
        x1,y1,x2,y2 = line[0]
    
    #draw the line based on two points 
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
  return image

def reduce_noise(image, kernel_size = 5):
  '''reduce noise of grayscale image using gausian blur'''
  return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def get_lines(masked_edge_image, rho, theta_coef, min_votes, min_line_length, max_line_gap):
  '''convert edges into lines using hough transform algorithm '''
  theta = theta_coef*np.pi/180
  return cv2.HoughLinesP(masked_edge_image, rho, theta, min_votes, np.array([]), minLineLength = min_line_length, maxLineGap = max_line_gap)



#def nothing(x):
#    pass


def get_edges(blur_image, low_threshold = 100, high_threshold = 200):
  '''find edges using canny transform algorithm'''
  return cv2.Canny(blur_image, low_threshold, high_threshold)

def move(line_l, line_r, pr):
  char = 'w'
  if line_r is None and line_l is not None:
    char = 'd'
    return char
  if line_l is None and line_r is not None:
    char = 'a'
    return char
  if not line_r is None and line_l is not None:
    if pr == 'a':
      char = 'a'
      return char
    if pr == 'd':
      char = 'd'
      return char
  return char

param = {

    #canny transform parameters
  'canny_lo': 100, 
  'canny_hi': 200, 

  #hough parameters
  'rho': 1, 
  'theta_coef': 1, 
  'min_votes': 30, 
  'min_line_length': 0, 
  'max_line_gap': 250
}

#cap = cv2.VideoCapture('pistavid2.mp4')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)
motor_init()
prev = 'w'

#cv2.namedWindow("Trackbars")
#cv2.namedWindow("Lanes")

#cv2.createTrackbar("L - H", "Trackbars", 0, 640, nothing)
#cv2.createTrackbar("L - S", "Trackbars", 0, 480, nothing)
#cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
#cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("Left Lane", "Lanes", 0, 255, nothing)
#cv2.createTrackbar("Right Lane", "Lanes", 0, 255, nothing)



while True:
    ret, frame = cap.read() 
    if not ret:
       cap = cv2.VideoCapture(0)
       cap.set(3,640)
       cap.set(4, 480)
       continue
    blur_image = reduce_noise(frame)
    hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)

 #   l_h = cv2.getTrackbarPos("L - H", "Trackbars")
  #  l_s = cv2.getTrackbarPos("L - S", "Trackbars")
  #  l_v = cv2.getTrackbarPos("L - V", "Trackbars")
  #  u_h = cv2.getTrackbarPos("U - H", "Trackbars")
  #  u_s = cv2.getTrackbarPos("U - S", "Trackbars")
  #  u_v = cv2.getTrackbarPos("U - V", "Trackbars")
  #  ll =  cv2.getTrackbarPos("Left Lane", "Lanes")
  #  rl =  cv2.getTrackbarPos("Right Lane", "Lanes")

    low_black = np.array([0, 0, 0])
    high_black = np.array([70, 255, 70])
    mask_hsv = cv2.inRange(hsv, low_black, high_black)

##    low_black = np.array([0, 0, 0])
##    high_black = np.array([179, 255, 100])
##    mask_hsv = cv2.inRange(hsv, low_black, high_black)
    edge_image = get_edges(mask_hsv)
    
    polygon = np.array([
    [(0,200), (200,200), (200,480), (0, 480)]
    ])
    maskl = np.zeros_like(edge_image)
    cv2.fillPoly(maskl, polygon, 255)
    left_mask = cv2.bitwise_and(edge_image, maskl)
    
    polygon = np.array([
    [(640,200), (440, 200), (340,480), (640, 480)]
    ])
    maskr = np.zeros_like(edge_image)
    cv2.fillPoly(maskr, polygon, 255)
    right_mask = cv2.bitwise_and(edge_image, maskr)




    right_lines = get_lines(right_mask, param['rho'], param['theta_coef'],
                           param['min_votes'], param['min_line_length'],
                           param['max_line_gap'])
    left_lines = get_lines(left_mask, param['rho'], param['theta_coef'],
                           param['min_votes'], param['min_line_length'],
                           param['max_line_gap'])

    kat = move(left_lines, right_lines, prev)
    kinhsh(kat)
    prev= kat
    left_line_image = extrapolate_lines(left_lines, frame)
    right_line_image = extrapolate_lines(right_lines, frame)

    cv2.imshow('frame', frame)
    #cv2.imshow('hsv', mask_hsv)
    #cv2.imshow('left_mask', left_mask)
    #cv2.imshow('right_mask', right_mask)


    
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()

