# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 14:13:01 2019

@author: ks4617
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
#%%                                                                Loading data
hdulist = fits.open("H:\Year 3 Labs\Astronomical\A1_mosaic\A1_mosaic.fits")
data = hdulist[0].data
hdulist.close()
#%%
def gauss(data,lowlim,upplim):                              # To find global background
    onedata = data.flatten()                                # Flatten to 1d array
    upp_limit = onedata<upplim                              
    bottom_half = onedata[upp_limit]                        # Cut data above upper limit set for background gaussian
    low_limit = bottom_half>lowlim
    final = bottom_half[low_limit]                          # Cut data below lower limit set for background gaussian
    n,bins,patches = plt.hist(final,10**3,edgecolor='None') # Plot histogram of background
    
    a = np.where(n == sorted(n)[-1])                        # Find index where the peak of gaussian is 
    mu = 0.5*(bins[a[0]]+bins[a[0]+1])                      # Calculate the mean looking at the midpoint of the peak bin
    mu=mu[0]
    
    half_max = 0.5*np.max(n)
    index2 = np.where(n>half_max)[0][0]                     # Index for bin to the right of half maximum
    y2 = n[index2]                                          # y value for bin to the right of half maximum 
    left_max = np.where((n>0)&(n<y2))[0]                    # Index for bin to the left of half maximum
    halfmax_lim = left_max<index2
    left_max2 = left_max[halfmax_lim]
    index1 = left_max2[-1]
    y1 = n[index1]                                          # y value for bin to the left of half maximum 
    x1 = 0.5*(bins[index1]+bins[index1+1])                  # x value for bin to the left of half maximum 
    x2 = 0.5*(bins[index2]+bins[index2+1])                  # x value for bin to the right of half maximum 
    lower_x = x1 + (x2-x1)*(half_max-y1)/(y2-y1)            # x value for FWHM
    FWHM = 2*(mu-lower_x)                                   
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))                 # Standard deviation
    
    x=np.linspace(lowlim,upplim,10000)
    y = np.empty(len(x))
    y.fill(half_max)
    G=(1/(sigma*np.sqrt(2*np.pi)))*np.e**(-((x-mu)/(np.sqrt(2)*sigma))**2)  # Gaussian equation
    G_scaled = G*np.max(n)/np.max(G)                                        # Scaled to histogram size
    plt.xlabel('Count number')
    plt.ylabel('Frequency')
    plt.plot(x,G_scaled,color='red',linewidth=2)                            # Plot Gaussian
    n,bins,patches = plt.hist(final,10**3,edgecolor='None',color='blue')    # Plot histogram
    plt.show()
    return mu, sigma
    
def count_source(data1,background):                 # To count number of sources
    cat = np.zeros((1,4))                           # Define catalogue array
    while np.max(data1)>background:                 # Loop through all sources above the global background
        coords = np.where(data1 == np.max(data1))   # Take the coordinates of the brightest pixel
        vert = coords[0][0]
        horz = coords[1][0]
        cat = np.append(cat, [[vert,horz,np.max(data1),1]],axis=0)  # Save coordinates and  brightness for the first pixel of the source
        data1[vert,horz] = 0                                        # Set the pixel found to 0 to mask it once recorded
        brightness = 0                                              # Brightness of the rest of the source defined as 0 to begin
       
        for n in range(1,7):
            for i in range(2*n+1):          
                if vert+n < np.shape(data1)[0] and horz-n+i < np.shape(data1)[1] and vert+n > -1 and horz-n+i > -1:      # Iterate through left top corner to right top corner
                    if data1[vert+n][horz-n+i]>background:          # If pixel brightness is greater than global background
                        brightness += data1[vert+n][horz-n+i]       # Add brightness of pixel to total brightness of source
                        data1[vert+n,horz-n+i] = 0                  # Mask recorded pixel
                        cat[-1][3] += 1                             # Add one to pixel number count for source

            for i in range(2*n+1):
                if vert+n-i < np.shape(data1)[0] and horz+n < np.shape(data1)[1] and vert+n-i > -1 and horz+n < np.shape(data1)[1]: # Iterate through right top corner to right bottom corner 
                    if data1[vert+n-i][horz+n]>background:
                        brightness += data1[vert+n-i][horz+n]
                        data1[vert+n-i,horz+n] = 0
                        cat[-1][3] += 1

            for i in range(2*n+1):
                if vert-n > -1 and horz+n-i < np.shape(data1)[1] and vert-n > -1 and horz+n-i > -1:                 # Iterate through right bottom corner to left bottom corner
                    if data1[vert-n][horz+n-i]>background:
                        brightness += data1[vert-n][horz+n-i]
                        data1[vert-n,horz+n-i]= 0

            for i in range(2*n+1):
                if vert-n+i > -1 and horz-n > -1 and vert-n+i < np.shape(data1)[0] and horz-n > -1:                 # Iterate through left bottom corner to right top corner
                    if data1[vert-n+i][horz-n]>background:
                        brightness += data1[vert-n+i][horz-n]
                        data1[vert-n+i,horz-n]= 0
                        cat[-1][3] += 1
 
        cat[-1][2] += brightness                    # Add brightness of whole source to catalogue 
        print(np.shape(cat))
    limit = cat[:,-1] >5
    cat = cat[:][limit]                             # All of catalogue where the number of pixels in source is greater 5
    cat=np.delete(cat,0,0)                          # Remove first place holder row
    print(np.shape(cat))
    plt.figure()
    plt.pcolormesh(data1)                           # Plot image with soruces masked
    plt.colorbar()
    plt.show()
    return cat
    
def mask_frame(data):                                                   # Initial mask of obvious frames around the image
    frame_mask = np.zeros(np.shape(data))
    a = np.where(data == 3421)                                          # All indices where the brightness is equal to 3421
    for i in range(len(a[0])):
            frame_mask[a[0][i]][a[1][i]] = data[a[0][i]][a[1][i]]       # Set all indices where data = 3421 to 3421 in frame mask
            
    j_valuesleft = []                                                   # Number of 3421 pixels from the left side for each row
    for i in range(np.shape(frame_mask)[0]):
        j = 0
        while frame_mask[i][j]>=3421 and j<np.shape(frame_mask)[1]-1:   # Count outwards from edge until frame stops
            j+=1
        j_valuesleft.append(j)
    
    j_valuesright = []                                                  # Number of 3421 pixels from the right side for each row
    for i in range(np.shape(frame_mask)[0]):
        j = 2569
        while frame_mask[i][j]>=3421 and j>=1:                          # Count outwards from edge until frame stops
            j-=1
        j_valuesright.append(j)
    
#    for i in range(np.shape(frame_mask)[0]):
#        for j in range(j_valuesleft[i]):
#            frame_mask[i][j] = 0
#            
#    for i in range(np.shape(frame_mask)[0]):
#        for j in range(j_valuesright[i],2570):
#            frame_mask[i][j] = 0
    
    for i in range(np.shape(frame_mask)[0]):                            # Mask all frame on the left side
        for j in range(j_valuesleft[i]):
            data[i][j] = 0
            
    for i in range(np.shape(frame_mask)[0]):                            # Mask all frame on the right side
        for j in range(j_valuesright[i],2570):
            data[i][j] = 0
    return data
    
def local_b(data,cat,background):                                  # Find local background for region around source
    for k in range(len(cat)):                           # Loop through sources
        w=50                                            # Width of aperture to find local background around source
       # aperture = np.array([0.])
        
        #x = int(cat[k][1]+w-cat[k][1]+w+1)
        #y = int(cat[k][0]+w-cat[k][0]+w+1)
        
        x = int(2*w+1)                                  # Dimensions of aperture
        y = int(2*w+1)
        
        left = cat[k][1]-w                              # x value to start from (top left corner)
        top = cat[k][0]+w                               # y value to start from (top left corner)
        
        if left < 0:                                    # If the x starting point is out of the image set the value to the edge of the image
            left = 0
        if top < 0:                                     # If the y starting point is out of the image set the value to the edge of the image
            top = 0
          
        if left+w*2+1 >= np.shape(data)[1]:             # If the x finishing point is out of the image set the value to the edge of the image
            left = np.shape(data)[1]-w*2-1

        if top+w*2+1 >= np.shape(data)[0]:              # If the y finishing point is out of the image set the value to the edge of the image
           top = np.shape(data)[0]-w*2-1
            
        
        aperture = np.zeros((y,x))
        for j in range(0,2*w+1):                        # Create region to check local background in
            for i in range(0,2*w+1):
                aperture[j][i] = data[int(top)+j][int(left)+i]
           
        
#        aperture_flat = aperture.flatten()              # Remove all source terms from aperture
#        source_limit = aperture_flat<background
#        back = aperture_flat[source_limit]
        
        back = aperture.flatten()
      
        non_zero = back>0                               # Remove all zero terms
        back = back[non_zero]
        
        mu,sigma = gauss(aperture,3200,3700)            # Calculate mu and sigma for  local background
#        mu,sigma = gauss(aperture,3380,3450)
#        mu = np.mean(aperture)
        cat[k][2] -= mu*cat[k][3]                       # Subtract local background
        return back, cat, aperture
        
def graph(cat):
    m = np.array([])
    for n in range(len(cat)):
        m = np.append(m,25.3 - 2.5*np.log10(cat[n][2]))
        
    m_sorted = np.asarray(sorted(m))
    N = np.arange(0,len(m_sorted))
    log_N = np.log10(N)
    
    errorN = np.sqrt(N)
    errorlogN = errorN/(np.log(10)*N)

    plt.figure() 
    plt.xlabel("m")
    plt.ylabel("log(N(<m))")
    plt.plot(m_sorted, np.log10(N),'x')
    plt.errorbar(m_sorted, np.log10(N),yerr=errorlogN,ls='none')
    m_cropped = m_sorted[np.where(m_sorted > 10.8)[0][0]:np.where(m_sorted>12.4)[0][0]]
    log_N_cropped = log_N[np.where(m_sorted > 10.8)[0][0]:np.where(m_sorted>12.4)[0][0]]
    errorlogN_c = errorlogN[np.where(m_sorted > 10.8)[0][0]:np.where(m_sorted>12.4)[0][0]]
    
    fit,cov = np.polyfit(m_cropped,log_N_cropped,1,w=1/errorlogN_c,cov=True)
    grad_err = np.sqrt(cov[0,0])
    intercept_err = np.sqrt(cov[1,1])
    print('gradient=',fit[0],'+-',grad_err)
    print('intercept=',fit[1],'+-',intercept_err)
    z = np.poly1d(fit)
    plt.plot(m_cropped ,z(m_cropped))
    plt.show()
    return

#%%                                                                  Test data
testdata = np.zeros((30,30))
testdata[7][7] = 5
testdata[7][6] = 2
testdata[7][8] = 3
testdata[10][10] = 1
testdata[5][9] = 1
testdata[14,14] = 1
testdata[29,29] = 1
testdata[29,28] = 5
testdata[5,24] = 6
testdata[6,25] = 4
testdata[7,25] = 3

cat_testdata1 = count_source(testdata,0)
back1, cat1, aperture1 = local_b(testdata,cat_testdata1)
#%%                                             Cropped part of whole image - 1
image = data[1040:1280,890:1150]
testimg = np.copy(image)

#%%
plt.figure()
plt.pcolormesh(testimg)
plt.colorbar()
plt.show()     

#%%
mu_test, sigma_test = gauss(testimg,3200,3700)
#%%
cat_testimg = count_source(testimg,mu_test+sigma_test*5)
#%%
back, cat, aperture = local_b(testimg,cat_testimg)  
#%%                                             Cropped part of whole image - 1
image = data[1690:2031,1750:2180]
testimg2 = np.copy(image)
        
plt.pcolormesh(testimg2)
plt.colorbar()
plt.show()
#%%
mu_test2, sigma_test2 = gauss(testimg2,3200,3700)
#%%
cat_testimg2 = count_source(testimg2,mu_test2+sigma_test2*5)
#%%
back2, cat2, aperture2 = local_b(testimg2,cat_testimg2) 
#%%
graph(cat_testimg2)
#%%                                                    Star mask for full data
data_masked = np.copy(data)
data_masked[0:len(data_masked),1370:1500] = 0

data_masked[3194:3430,750:795] = 0
data_masked[2695:2850,960:995] = 0
data_masked[2214:2352,890:925] = 0
data_masked[0:25,962:1724] = 0
data_masked[116:147,1305:1550] = 0
data_masked[314:336,1014:1713] = 0
data_masked[423:465,1091:1659] = 0
data_masked[3705:3806,2120:2147] = 0
data_masked[3840:3853,2272:2285] = 0
data_masked[3380:3450,2455:2475] = 0
data_masked[4080:4120,545:575] = 0
data_masked[2280:2340,2122:2145] = 0
data_masked[1395:1455,2075:2100] = 0
data_masked[4325:4340,1358:1370] = 0
data_masked[4385:4405,1300:1325] = 0
data_masked[25:45,1636:1642] = 0
data_masked[25:50,1680:1705] = 0
data_masked[565:590,1760:1785] = 0
data_masked[4265:4283,48:62] = 0
data_masked[3365:3380,888:902] = 0
data_masked[2285:2310,435:455] = 0
data_masked[1480:1510,625:645] = 0
data_masked[1925:1950,665:685] = 0
data_masked[2250:2270,695:715] = 0
data_masked[3285:3305,2250:2270] = 0
data_masked[1730:1750,676:700] = 0
data_masked[3010:3050,2500:2535] = 0
data_masked[1100:1135,2505:2535] = 0
data_masked[280:295,1790:1805] = 0
data_masked[3915:3930,180:195] = 0
data_masked[2500:2540,2520:2545] = 0
data_masked[1645:1665,960:972] = 0
data_masked[650:675,45:85] = 0
data_masked[4310:4320,570:585] = 0
data_masked[835:850,201:216] = 0
data_masked[3200:3210,845:860] = 0
data_masked[2900:3500,1100:1800] = 0
data_masked[3150:3450,720:840]= 0
data_masked[3700:3800,2100:2170]= 0
data_masked[2650:2850,930:1030]= 0
data_masked[2210:2360,860:950]= 0
data_masked[1370:1470,2060:2130]= 0
data_masked[3200:3350,2150:2350]= 0
data_masked[4060:4130,510:600]= 0
data_masked[1450:1520,610:660]= 0
data_masked[4400:4500,2150:2500]= 0
data_masked[4200:4500,2350:2500]= 0
data_masked[0:len(data),2470:np.shape(data)[1]]= 0
data_masked[0:150,0:np.shape(data)[1]]= 0
data_masked[0:200,0:450]= 0
data_masked[0:400,0:200]= 0
data_masked[0:len(data),0:100]= 0
data_masked[550:750,0:140]= 0
data_masked[805:870,180:280]= 0
data_masked[4500:len(data),0:np.shape(data)[1]]= 0
data_masked[2270:2337,2100:2160]= 0
data_masked[3360:3470,2420:2470]= 0
data_masked[550:600,1750:1793]= 0
data_masked[1550:1570,1925:1940]= 0
plt.figure()
plt.pcolormesh(data_masked)
#%%
mask_frame(data_masked)
mu_data, sigma_data = gauss(data_masked,3200,3700)
#%%
cat_data = count_source(data_masked,mu_data+sigma_data*5)
#%%
backd, catd, apertured = local_b(data_masked,cat_data,mu_data+sigma_data*5)
#%%
graph(catd)

