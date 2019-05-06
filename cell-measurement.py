import cv2
import os
import numpy as np

def nothing(x):
    pass

def RemoveCell(event,x,y,flags,param):
    if remove_flag is True:
        if event == cv2.EVENT_RBUTTONDOWN:
            #print('1')
            for i,conti in enumerate(cell_contours_auto_removal):
                if cv2.pointPolygonTest(conti,(x,y),False) >= 0:
                    remove_manual_list.append(i)
        if event == cv2.EVENT_MBUTTONDOWN and len(remove_manual_list) > 0:
            del remove_manual_list[-1]

def DetectScaleBar():
    _,black = cv2.threshold(img,10,255,cv2.THRESH_BINARY_INV)
    _,white = cv2.threshold(img,245,255,cv2.THRESH_BINARY)
    _,blackbar,_ = cv2.findContours(black,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    _,whitebar,_ = cv2.findContours(white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #print (bar)
    for bari in blackbar:
        x,y,w,h = cv2.boundingRect(bari)
        area = cv2.contourArea(bari)
        if area > 0 and w * h / area <1.25 and w / h > 5:
            return (w)
    for bari in whitebar:
        x,y,w,h = cv2.boundingRect(bari)
        area = cv2.contourArea(bari)
        if area > 0 and w * h / area <1.25 and w / h > 5:
            return (w)
    return (0)

def ExportData():
    file = open('./results/' + imgfile + '.txt','w')
    file.write("scalebar = %i \n\n" % (scalebar_length))
    file.write("Index\tArea\tCircularity\n")
    for i in range(len(area_list)):
        file.write("%i\t%.1f\t%.3f\n" % (i,area_list[i],circularity_list[i]))
    file.close()

remove_flag = False
if os.path.isdir('./results') == False:
    os.mkdir('./results')
for imgfile in os.listdir('.'):
    if imgfile.endswith('.jpg') or imgfile.endswith('.bmp'):

        cv2.namedWindow('cells',cv2.WINDOW_NORMAL)
        cv2.createTrackbar('MAGIC_PARA', 'cells', 10, 20, nothing)
        cv2.createTrackbar('FOCUS', 'cells', 4, 10, nothing)
        cv2.createTrackbar('MIN_SIZE', 'cells', 1, 10, nothing)
        cv2.setMouseCallback('cells', RemoveCell)

        img0 = cv2.imread(imgfile)
        img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
        (height,width) = img.shape
        resize_factor = height / 1000
        scalebar_length = DetectScaleBar()
        mblur = cv2.medianBlur(img,5)
        blur = cv2.GaussianBlur(mblur,(5,5),0)

        e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        canvas = np.zeros(img.shape,np.uint8)


        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
        edges = (sobelx ** 2 + sobely ** 2) ** 0.5
        edges = (edges/np.max(edges)*255).astype(np.uint8)
        #_,edges = cv2.threshold(edges,30,255,cv2.THRESH_BINARY)
        blocksize = int(img.shape[0] / 30) + int(img.shape[0] / 30 + 1) % 2

        MIN_SIZE = -1
        MAGIC_PARA = -1
        FOCUS = -1
        remove_flag = False

        while(1):
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
            elif k == 32:
                remove_flag = not remove_flag
            min_size = cv2.getTrackbarPos('MIN_SIZE','cells')
            magic_para = cv2.getTrackbarPos('MAGIC_PARA','cells')
            focus = cv2.getTrackbarPos('FOCUS','cells')
            im = np.copy(img0)
            #print (remove_flag)
            if MIN_SIZE != min_size or MAGIC_PARA != magic_para or focus != FOCUS:
                remove_flag = False
                remove_manual_list = []
                MIN_SIZE = min_size
                MAGIC_PARA = magic_para
                FOCUS = focus

                edges_thresh = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,\
                                              blocksize,-5-MAGIC_PARA)

                _,contours,_ = cv2.findContours(edges_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                img_thresh = np.copy(canvas)

                for conti in contours:
                    hull = cv2.convexHull(conti)
                    cv2.drawContours(img_thresh,[hull],0,255,-1)
                img_thresh = cv2.morphologyEx(img_thresh,cv2.MORPH_OPEN,e_kernel,iterations=1)
                #img_thresh = cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,d_kernel,iterations=1)
                #cv2.imshow('1',cv2.erode(img_thresh,e_kernel,iterations=3))
                _,cell_contours,_ = cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                remove_auto_list = []
                for i,conti in enumerate(cell_contours):
                    size = cv2.contourArea(conti)
                    if size ** 0.5 < MIN_SIZE * height / 200:
                        remove_auto_list.append(i)
                    else:
                        celli = np.copy(canvas)
                        cv2.drawContours(celli, [conti], 0, 255, -1)
                        x,y,w,h = cv2.boundingRect(conti)
                        #cv2.imshow('roi',celli[y:y+h,x:x+w])
                        #periphery_mean_intensity = cv2.mean(blur[y:y+h,x:x+w],mask=cv2.erode(celli[y:y+h,x:x+w],e_kernel,iterations=1))[0]
                        #center_mean_intensity = cv2.mean(blur[y:y+h,x:x+w],mask=cv2.erode(celli[y:y+h,x:x+w],e_kernel,iterations=2))[0]
                        lap = cv2.Laplacian(img[y:y + h, x:x + w], cv2.CV_8U)
                        #print(lap)
                        lap_mean = cv2.mean(lap,mask=celli[y:y+h,x:x+w])[0]
                        lap_var = cv2.mean((lap - lap_mean) ** 2,mask=celli[y:y+h,x:x+w])[0]
                        #print(lap_var)
                        if lap_var < FOCUS * 10:
                            #if periphery_mean_intensity > center_mean_intensity:
                            remove_auto_list.append(i)
                if len(cell_contours) == 1 and len(remove_auto_list) == 1:
                    cell_contours_auto_removal = []
                else:       
                    cell_contours_auto_removal = np.delete(cell_contours, remove_auto_list)
                
            if len(cell_contours_auto_removal) == 1 and len(remove_manual_list) == 1:
                cell_contours_manual_removal = []
            else:
                cell_contours_manual_removal = np.delete(cell_contours_auto_removal,remove_manual_list)
            cv2.drawContours(im, cell_contours_manual_removal, -1, (0, 255, 0), 2)
            #im_resized = cv2.resize(im,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
            #im_resized = cv2.resize(im, (800,800), interpolation=cv2.INTER_AREA)
            #edges_resized = cv2.resize(edges,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
            #cv2.imshow('edges',edges)
            cv2.imshow('cells',im)
            cv2.resizeWindow('cells',int(width / resize_factor),1000)

        area_list = []
        circularity_list = []
        for i,conti in enumerate(cell_contours_manual_removal):
            area = cv2.contourArea(conti)
            (x,y),radius = cv2.minEnclosingCircle(conti)
            cv2.putText(im,str(i),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
            enclosing_area = np.pi * radius ** 2
            area_list.append(area)
            circularity_list.append(area / enclosing_area)

        ExportData()
        cv2.imwrite('./results/' + imgfile,im)
        cv2.destroyAllWindows()
