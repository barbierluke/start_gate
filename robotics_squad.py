#!/usr/bin/env python
from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math
from operator import itemgetter
from pdb import set_trace

calibration = 1600 # distance = calibration * pixels^-1
# we want inverse relation so as pixels increase, distance lessens

num = 0
folder = 'images/'
START_GATE_LEG_PIXELS = 30
FRAME_WIDTH = 752

# for use in getBiModalMeans()
BUCKET_START_IND = 0
NUM_POINTS_IND = 1
POINT_LIST_IND = 2

def getBiModalMeans(pts):
    # here we group the neightboring points in order to establish where the start gate legs are located. We do calculate buckets twice in an overlapping method to account for the case in which we happen to split nearby points on the same leg into two buckets
    buckets = []
    for i in range(0,FRAME_WIDTH, START_GATE_LEG_PIXELS):
        bucket_pts = [x for x in pts if (x >= i and x <= i + START_GATE_LEG_PIXELS)]
        buckets.append([i, len(bucket_pts), bucket_pts])
    for j in range(int(START_GATE_LEG_PIXELS/2), FRAME_WIDTH, START_GATE_LEG_PIXELS):
        bucket_pts = [x for x in pts if (x >= i and x <= i + START_GATE_LEG_PIXELS)]
        buckets.append([j, len(bucket_pts), bucket_pts])
    # we have buckets
#    print buckets
    buckets_sorted = list(reversed(sorted(buckets, key=itemgetter(NUM_POINTS_IND))))
#    print "-------------"
#    print buckets_sorted

    first_pole = buckets_sorted[0] # first bucket 
    i = 1 # start at the 2nd highest bucket
    # find the 2nd highest bucket that's not a part of the first highest's leg
    while not (buckets_sorted[i][BUCKET_START_IND] >= first_pole[BUCKET_START_IND] + 3* START_GATE_LEG_PIXELS or buckets_sorted[i][BUCKET_START_IND] <= first_pole[BUCKET_START_IND] - 3*START_GATE_LEG_PIXELS):
        i += 1
    second_pole = buckets_sorted[i] # we exited the while on the condition that this bucket was not on the same leg as the previously higher buckets
    
    if len(second_pole[2]) < 1: # check if we didn't find the 2nd leg
        return -1,-1
    first_mean = np.mean(first_pole[2])
    second_mean = np.mean(second_pole[2])
    print int(first_mean)
    print int(second_mean)
    return first_mean, second_mean

def printRectangle(array, topLeft=[0,0], bottomRight=[1,1]):
    for row in range(topLeft[0], bottomRight[0]+1):
        print str(row)+": " + str(array[row][topLeft[1]:bottomRight[1]])

def findPole(array, point, r=20, delta=10): # delta is our acceptable range
    # WE APPROACH FROM THE RIHGT
    sections = []
    originalFound = False # did we find the water again
    row = point[0]
    sections.append([array[row][point[1]+r-1], 
                     array[row][point[1]+r-2], 
                     array[row][point[1]+r-3],
                     array[row][point[1]+r-4],
                     array[row][point[1]+r-5]]) # append first 3 points (we assume this is the water distribution
    i = point[1]+r - 6
    sections.append([]) # initialize poll points
    switching = False
    foundOtherWater = False
    rightIndex = 0
    while not foundOtherWater: # did we find water on the other side of the poll?
        u = np.mean(sections[0])
        print "i: "+str(i) + " u: " + str(u) +" [" +str(row)+ ","+ str(i) + "] val: " + str(array[row][i])
        if abs(array[row][i] - u) < delta:
            sections[0].append(array[row][i])
            if switching == True:
                foundOtherWater = True
        else:
            if len(sections[1]) == 0: # this is our first point
                rightIndex = i
            sections[1].append(array[row][i])
            switching=True
            print "not water"            
        i -= 1
        print "sections: " + str(sections)
    print "Found other side baby at index: " + str(i)
    return sections[0], sections[1], i, rightIndex

def bottomOut(array,row, startIndex, polePointsR2L, waterPointsR2L, delta=10):
    # we'll create a potential third distribution, we want to know when we enter the water again
    # head south

    # reverse the lists
    polePoints = list(reversed(polePointsR2L))
    waterPoints = list(reversed(waterPointsR2L))
    blackPoints = []
    foundWater = False # if we can't go left or right more than 3 indices, we've hit the bottom
    col = startIndex
    row -= 1
    uWater = np.mean(waterPoints)
    moved = 0
    print "looping"
    while not foundWater:
        # first see if we can go downward, if not head inward by three points
        if abs(array[row][col] - uWater) < delta: # we've got water
            array[row][col] = 255; # set to white
            col += 1
            moved += 1
            print "moving over"
            if moved > 2:
                print "moved 3 times: at point: "+ str(row) + "," + str(col)+"]"
                break
        elif abs(array[row][col] - np.mean(polePoints)) < delta:
            polePoints.append(array[row][col])
            array[row][col] = 255; # set to white
            moved = 0
            row += 1
            print "moving down, row: " + str(row)
        elif array[row][col] < np.mean(polePoints): # we should be darker...
            array[row][col] = 255; # set to white
            moved = 0
            print "new distribution at point: [" + str(row) + "," + str(col)+"] "+ str(array[row][col])
            row += 1
        else:
            print "v unexpected event... at: [" + str(row) + "," + str(col)+"] " + str(array[row][col])
            # don't know why this would be... hope everything is fine!
            array[row][col] = 255; # set to white
            row += 1 
            col += 1
            moved += 1
            if moved > 3:
                print "moved 3 times: at point: "+ str(row) + "," + str(col)+"]"
                break
        
    return row, col

def climbLeg(array, row, centerCol, width, waterPoints, polePoints, delta=10):
    """
    We assume wef are given a row and a centerCol that are in the tape section of the pole.
    I.e. not black, not water.
    We also assume that polePoints are not black. From bottomOut
    We also pass in width, which is the result of findPole from somewhere else.
    """
    # we are given center
    # go up

    poleMean = np.mean(polePoints)
    waterMean = np.mean(waterPoints)
    col = centerCol
    while row >= 0:
        if abs(array[row][col] - poleMean) < delta:
            #means we have another polePoint
            polePoints.append(array[row][col])
            #update poleMean
            poleMean = np.mean(polePoints)
            # decrement row
            array[row][col] = 255; # set to white
            
        elif abs(array[row][col] - waterMean) < delta:
            #means we have water
            waterPoints.append(array[row][col])
            waterMean = np.mean(waterPoints)
            
            new_col = butterflyOut(array, row, col, width, polePoints, delta, True)
            if new_col == None:
                # Not supposed to happen.
                # we're fucked
                print "DUNNO WHUT HAPPENT FAM!!!!"
                print "WE'RE DROWNING OVER HERE"
                print "Check if we are at black?"
            else:
                array[row][col] = 255; # set to white
                col = new_col
                        
        elif array[row][col] < poleMean:
            #means we have a black point
            new_col = butterflyOut(array, row, col, width, polePoints, delta, True)
            if new_col == None:
                # In the black baby
                # exit case
                break
            else:
                array[row][col] = 255; # set to white
                col = new_col

        row -= 1

    return row, col, polePoints, waterPoints

            
def butterflyOut(array, row, col, width, polePoints, delta, dir_up, _tries=0):
    """
    Looks both left and right until we find a polePoint
    Returns good col
    Tries a couple cols if failure (up cols if dir_up==TRUE, down cols else)
    Returns None if we failed
    """
    tries = _tries
    poleMean = np.mean(polePoints)
    waterMean = np.mean(waterPoints)
    print delta
    for inc in range(1,width-1):
        if abs(array[row][col+inc] - poleMean) < delta:
            # we have polePoint right
            return col+inc
        elif abs(array[row][col-inc] - poleMean) < delta:
            # we have poolePoint left
            return col-inc

    # means the row is fucked
    if tries > 2:
        return None
    else:
        if dir_up:
            return butterflyOut(array, row-1, col, width, polePoints, delta, dir_up, tries+1)
        else:
            return butterflyOut(array, row+1, col, width, polePoints, delta, dir_up, tries+1)
            

for name in os.listdir(folder):
    # if name != 'frame0105.jpg':
    #     continue
    img = cv2.imread(folder + name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,20,3)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    coords = []
    num_lines = 0
    for i in lines:
        for rho,theta in i:
            if not ( (-.2 <= theta <= .2) or (theta > 3)):
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
#            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            if x1 == x2:
                x2 += 1
            slope = (y2 - y1) / (x2 - x1)
            a2 = int((y1 - 240)//slope)
            x240 = x1-a2
            coords.append(x240)             
#            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
#            cv2.circle(img, (x1-a2,240), 3, (0,255,0), -1)

    pole1, pole2 = getBiModalMeans(coords)
    if pole1 < 0: # error
        print "ERROR !!!!!!!"
        print name
        continue
    pole1 = int(pole1)
    pole2= int(pole2)
    if pole2 < pole1: # make pole 1 on the left
        pole1, pole2 = pole2, pole1
    print type(gray)
    print gray.shape
    print pole1
    print "before"
    print gray[240][pole1] # access row, column
    cv2.circle(img, (pole1,240), 5, (0,255,0), -1)
    cv2.circle(img, (pole2,240), 5, (0,255,0), -1)
#    cv2.circle(gray, (pole1,240), 1, (0,255,0), -1)
#    cv2.circle(gray, (pole2,240), 3, (0,255,0), -1)
    print "after"
    print gray[240][pole1]

    #IMAGE 0098

    # Move left, move right until find the width of the pole (hopefully it put us on the pole, we may not be on the pole... (we're just somewhere near the pole...)
    r = 30

    # cv2.circle(gray, (pole1,240), 1, (0,255,0), -1)
    # cv2.circle(gray, (pole1-r,240), 1, (0,255,0), -1)
    # cv2.circle(gray, (pole1+r,240), 1, (0,255,0), -1)
    # printRectangle(gray, [240-r,pole1-r], [240+r,pole1+r])
    # cv2.imshow(str(num) + "_" +name, gray)
    # cv2.waitKey(0)
    
    print "-----------------"

    #### LEFT LEG BOTTOM 
    waterPoints, polePoints, leftIndex, rightIndex = findPole(gray, [240,pole1], r)
    midCol = (rightIndex + leftIndex) // 2 
    width = rightIndex - leftIndex
    print "midCol: {}, right: {}, left: {}".format(midCol, rightIndex, leftIndex)
#    set_trace()
    row, col = bottomOut(gray,240,midCol, polePoints, waterPoints)
    
    # find center of bottom
    throwAway1, throwAway2, leftIndex, rightIndex = findPole(gray, [row,col], r) # this should give us our left and right indices
    midCol2 = (rightIndex + leftIndex) // 2
    lower_left_money = [row, midCol2]

  
    ### LEFT LEG TOP
    
    top_l_row, top_l_col, polePoints, waterPoints = climbLeg(gray, 240, midCol, width, waterPoints, polePoints)
    trash1, trash2, leftIndex, trash3 = findPole(gray, [top_l_row,top_l_col], r)
    upper_left_money = [top_l_row, leftIndex]

    #### RIGHT LEG BOTTOM

    waterPoints, polePoints, leftIndex, rightIndex = findPole(gray, [240,pole2], r)
    midCol = (rightIndex + leftIndex) // 2
    width = rightIndex - leftIndex
    print "midCol: {}, right: {}, left: {}".format(midCol, rightIndex, leftIndex)
#    set_trace()
    row, col = bottomOut(gray,240,leftIndex, polePoints, waterPoints)
    waterPoints, polePoints, leftIndex, rightIndex = findPole(gray, [row,col+4], r) # this should give us our left and right indices
    midCol2 = (rightIndex + leftIndex) // 2
    lower_right_money = [row, midCol2]
    print "midCol: {}, right: {}, left: {}".format(midCol, rightIndex, leftIndex)

  
    
    ### RIGHT LEG TOP
    top_r_row, top_r_col, polePoints, waterPoints = climbLeg(gray, 240, midCol, width, waterPoints, polePoints)
    trash1, trash2, trash3, rightIndex = findPole(gray, [top_r_row,top_r_col], r)
    upper_right_money = [top_r_row, rightIndex]
    
    # draw priors
    cv2.circle(gray, (pole1,240), 3, (0,255,0), -1)
    cv2.circle(gray, (pole2,240), 3, (0,255,0), -1)

    #draw money
    lr = tuple(reversed(lower_right_money))#reverse before plotting
    ll = tuple(reversed(lower_left_money))
    ur = tuple(reversed(upper_right_money))
    ul = tuple(reversed(upper_left_money))
     
    cv2.circle(gray, lr, 10, (0,255,0), -1)
    cv2.circle(gray, ll, 10, (0,255,0), -1)
    cv2.circle(gray, ur, 10, (0,255,0), -1)
    cv2.circle(gray, ul, 10, (0,255,0), -1)


#    cv2.imshow(str(num) + "__" +name, edges)
    cv2.imshow(str(num) + "___" +name, img)
    cv2.imshow(str(num) + "_" +name, gray)    
    cv2.waitKey(0)
    
    
    print "################################################################"


total = len(os.listdir(folder))
print "Total: " + str(total)
print "Num successful: " + str(num)
