
import fitz
import os
#import win32com.client
from os import mkdir
from os.path import isdir
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract
import csv
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def connected_components_stats_demo(src):
    #src = cv2.GaussianBlur(src, (3, 3), 0)
    #gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)

    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))

    colors[0] = (0, 0, 0)
    image = np.copy(src)
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        cx, cy = centers[t]
        cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 255), 2, 8, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), colors[t], 1, 8, 0)
        cv2.putText(image,  str(t), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1);
        print("label index %d, area of the label : %d"%(t, area))

    cv2.imshow("colored labels", image)
    #cv2.imwrite("D:/labels.png", image)
    print("total component : ", num_labels - 1)
    

def check_folder(path):
    if not isdir(path):
        mkdir(path)
    
def plt_img(img):    
    plt.imshow(img)            
    plt.show()   
            
def pdf_to_png(pdf_folder , png_folder):
    pdffilelist = os.listdir(pdf_folder)
    #pdffile = glob.glob(filename)[0]
    for file in pdffilelist:
        if file.endswith(".pdf"):
            try:
                #print(file)
                pdf_file = os.path.join(pdf_folder , file)
                doc = fitz.open(pdf_file)
                for pg in range(len(doc)):
                    page = doc[pg]
                    zoom = int(800)
                    rotate = int(0)
                    trans = fitz.Matrix(zoom / 100.0 ,zoom / 100.0).preRotate(rotate)
                    pm = page.getPixmap(matrix = trans , alpha=0)
                    pm.save(os.path.join(png_folder , file[:-4] + "_" + str(pg) + ".png"))
                    #print('saved')
            except:
                print(file)
                print('error')  
                
def read_img(path):
    img = cv2.imread(path , 0)
    #img_resize = cv2.resize( img , ( w* 4 , h * 2) , cv2.INTER_CUBIC) 
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    return th
                
def remove_small_area(file , area):
    nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(file, connectivity=8)
    for i in range(nlabel):
        x,y,wig,high,area=stats[i]
        if (stats[i, cv2.CC_STAT_AREA] < area):
            file[labels == i] = 0
    return file
      
#需傳入二值圖      
def get_hor_ver_line(file):
    #horizontal line
    scale_e = 25
    kernel_row_e = cv2.getStructuringElement(cv2.MORPH_RECT,( int (w // scale_e) , 1))#cols//scale
    erode_row = cv2.erode(file,kernel_row_e,iterations = 1)
    erode_row = remove_small_area(erode_row  , 10)
    kernel_row_d = cv2.getStructuringElement(cv2.MORPH_RECT,( int (w // scale_e) + 1 , 3))
    dilate_row = cv2.dilate(erode_row,kernel_row_d,iterations = 1)
    #vertical line
    scale_d = 10
    kernel_col_d = cv2.getStructuringElement(cv2.MORPH_RECT,(1 ,int( h // scale_d) ))#rows//scale
    erode_col = cv2.erode(th,kernel_col_d,iterations = 1)
    #plt_img(erode_col)
    erode_col = remove_small_area(erode_col , 10)
    kernel_col_d = cv2.getStructuringElement(cv2.MORPH_RECT,(3 ,int( h // scale_d) + 1))
    dilate_col = cv2.dilate(erode_col,kernel_col_d,iterations = 1)         
    #merge
    merge = cv2.add(dilate_row,dilate_col)
    return merge 

def OCR(file,psm):
    text0 = pytesseract.image_to_string(file, lang="eng",config = '--psm ' + str(psm))
    title_texts = []
    title_texts.append(text0)
    print(title_texts)
    return title_texts

###閾值之後可用整張圖之長寬佔比計算
#放棄最後一個components,應該有更好的方法
def remove_noise(img):
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    #print(num_labels)
    #image = np.copy(src)
    temp_horizontal = []
    keep_horizontal = []
    for t in range(1, num_labels-1, 1):
        x, y, w, h, area = stats[t]
        cx, cy = centers[t]
        #horizontal resistance
        if w > 80 and h > 10:
            binary[labels == t] = 0
        #vertical resistance
        elif h > 80  and w > 10:
            binary[labels == t] = 0
        #isolated vertical line
        elif h//w > 5  and abs(stats[t, cv2.CC_STAT_LEFT] - stats[(t-1), cv2.CC_STAT_LEFT]) > 25 and abs(stats[t+1, cv2.CC_STAT_LEFT] - stats[(t), cv2.CC_STAT_LEFT]) > 25:
            binary[labels == t] = 0
        #isolated horizontal line
        elif w//h > 5  and abs(stats[t, cv2.CC_STAT_LEFT] - 
                                stats[(t-1), cv2.CC_STAT_LEFT]) > 20 and abs(stats[t+1, cv2.CC_STAT_LEFT] - stats[(t), cv2.CC_STAT_LEFT]) > 20 and abs(stats[t+1, cv2.CC_STAT_TOP] - stats[(t), cv2.CC_STAT_TOP]) > 30 and abs(stats[t+1, cv2.CC_STAT_TOP] - stats[(t), cv2.CC_STAT_TOP]) > 30  :
            """
            temp_horizontal.append(t)
            for i in range(len(temp_horizontal)):
                for j in range(i):
                    if abs(cx[temp_horizontal[i]] - cx[temp_horizontal[j]]) < 2:
                        keep_horizontal.append(temp_horizontal[i])
                        keep_horizontal.append(temp_horizontal[j])
          
            drop_list = list(set(temp_horizontal) - set(keep_horizontal))            
            print(drop_list)
            """
            binary[labels == t] = 0
            
        elif area < 20  and abs(stats[t, cv2.CC_STAT_LEFT] - stats[(t-1), cv2.CC_STAT_LEFT]) > 20 and abs(stats[t+1, cv2.CC_STAT_LEFT] - stats[(t), cv2.CC_STAT_LEFT]) > 20 :
            #從X軸Y軸位置保留:、:、_類型符號
            temp_horizontal.append(t)
            print(temp_horizontal)
            for i in range(len(temp_horizontal)):
                for j in range(i):
                    if abs(centers[temp_horizontal[i]][1] - centers[temp_horizontal[j]][1]) < 4 or abs(centers[temp_horizontal[i]][0] - centers[temp_horizontal[j]][0]) < 4:
                        keep_horizontal.append(temp_horizontal[i])
                        keep_horizontal.append(temp_horizontal[j])
            print(temp_horizontal)
            print(keep_horizontal)
        
            drop_list = list(set(temp_horizontal) - set(keep_horizontal))            
            print(drop_list)
            for l in range(len(drop_list)):
                binary[labels == l] = 0
            #binary[labels == t] = 0        
        elif (w < 5 and h < 5) and (abs(stats[t, cv2.CC_STAT_LEFT] - stats[(t-1), cv2.CC_STAT_LEFT]) > 20 or abs(stats[t+1, cv2.CC_STAT_LEFT] - stats[(t), cv2.CC_STAT_LEFT]) > 20) :
            #框出符合條件的component
            """colors = []
            for i in range(num_labels):
                b = np.random.randint(0, 256)
                g = np.random.randint(0, 256)
                r = np.random.randint(0, 256)
                colors.append((b, g, r))

            colors[0] = (0, 0, 0)
            cv2.circle(binary, (np.int32(cx), np.int32(cy)), 2, (0, 255, 255), 2, 8, 0)
            cv2.rectangle(binary, (x, y), (x+w, y+h), colors[t], 1, 8, 0)
            cv2.putText(binary, str(t), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1);
            print("label index %d, area of the label : %d"%(t, area))
            cv2.imshow("colored labels", binary)"""
            binary[labels == t] = 0
             
            """
        elif  w < 7 and h < 7 and abs(stats[t, cv2.CC_STAT_LEFT] - stats[(t-1), cv2.CC_STAT_LEFT]) > 20 and abs(stats[t+1, cv2.CC_STAT_LEFT] - stats[(t), cv2.CC_STAT_LEFT]) > 20 and abs(stats[t+1, cv2.CC_STAT_TOP] - stats[(t), cv2.CC_STAT_TOP]) > 30 and abs(stats[t+1, cv2.CC_STAT_TOP] - stats[(t), cv2.CC_STAT_TOP]) > 30:   
            #print(centers[t][0])
            #binary[labels == t] = 0
            
            temp_horizontal.append(t)
            for i in range(len(temp_horizontal)):
                for j in range(i):
                    if abs(centers[temp_horizontal[i]][1] - centers[temp_horizontal[j]][1]) < 2:
                        keep_horizontal.append(temp_horizontal[i])
                        keep_horizontal.append(temp_horizontal[j])
        
            drop_list = list(set(temp_horizontal) - set(keep_horizontal))            
            #print(drop_list)
            for t in range(len(drop_list)):
                binary[labels == t] = 0"""
            
    return binary
        
if __name__ == '__main__':
    """
    pdf_folder = "C:/Users/lulu/wistron/design_schematic_capture/"
    check_folder(pdf_folder)
    png_folder = "C:/Users/lulu/wistron/design_schematic_capture/png/"
    check_folder(png_folder)
    pdf_to_png(pdf_folder , png_folder)
    """
    th = read_img("C:/Users/lulu/wistron/design_schematic_capture/test/test_8.png")
    
    (h,w) = th.shape
    #plt_img(th)
    hori_and_verti = get_hor_ver_line(th)
    #plt_img(hori_and_verti)
    img_text =  th - hori_and_verti
    plt_img(img_text)
    
    img_text = remove_noise(img_text)
    plt_img(img_text)
    
    #img_cc = connected_components_stats_demo(th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    text = OCR(~img_text , psm = 11)


