#!/usr/bin/env python
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math

calibration = 1600 # distance = calibration * pixels^-1
# we want inverse relation so as pixels increase, distance lessens

num = 0
folder = 'images/'

for name in os.listdir(folder):
#    print(i)
    img = cv2.imread(folder + name)
#    height, width = img.shape[:2]
    # print(height)
    # print(width)
    # print("--------")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,20,3)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    coords = []
    num_lines = 0
    for i in lines:
        for rho,theta in i:
            if not ( (-.05 <= theta <= .05) or (theta > 3)):
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            num_lines += 1
            color = (num_lines*30)%255
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            if x1 == x2:
                x2 += 1
            slope = (y2 - y1) / (x2 - x1)
            a2 = int((y1 - 240)//slope)
            x240 = x1-a2

            
            # if len(coords) != 0:
            #     if abs(coords[0] - x240) < 300: # not enough distance between lines
            #         continue
            #     if len(coords) > 1:
            #         break
            coords.append(x240)             
#            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.circle(img, (x1-a2,240), 3, (0,255,0), -1)

            # Now it's time to generate our 4 points
            """
            First we start at the mid coord and go left and right on the line within 3 pixels until we locate an orange or black color
            Now that I'm at the middle of the frame, I should simultaneous go left and right until I find a more orange than blue color
            """
            
    # if len(coords) < 2: # we didn't get enough hits
    #     print("not enough hits")
    #     continue
    # dist = abs(coords[0] - coords[1])
    # theta = (752//2 - (coords[0] + coords[1])//2)
    # print "coords: " + str(coords)
    # print "Dist: " + str(dist)
    # print "theta: " + str(theta)

    # estDist = calibration * (1/dist)
    # estTheta = math.atan2(theta, dist)
    # print "estDist: " + str(estDist)
    # print "estTheta: " + str(estTheta)

    # relX = estDist * math.cos(estTheta)
    # relY = -1*(estDist * math.sin(estTheta)) # flip the sign
    # print "relCoord: " + str([relX, relY])
    # num += 1
    print num_lines
    cv2.imshow(str(num) + "_" +name, img)
    cv2.waitKey(0)

total = len(os.listdir(folder))
print "Total: " + str(total)
print "Num successful: " + str(num)


"""
Next steps are subscribe to what /occam/image0 and publish another image with the start gate marked

"""
