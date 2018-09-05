# stunning-color

Dominant color extraction and colorspace mapping

Demo of a dominant color extractor.  Think of this as the automatic process your eyes do to figure out what color an object is.  We say that object is red or blue. A computer executes this by looking for the largest grouping of a single color in an object.  

This code is a collection of background extractors and a disabled skin detector that returns a dominant color for the input image and then maps that color to a static list of color supplied beforehand for naming.

Execute with to see examble on detecting color of a blue leather jacket: 

$ python mapmycolordetector.py https://d3ecedzw1mv51p.cloudfront.net/2017/10/Ladies-Leather-Biker-Jacket-Blue-Sue.jpg

Code will save off 4 intermediate images to show each phase of the background extraction and final evaluated image without a background for reference.

Requirements:

Code is all python3 and organized sequentially for easy of interpretibility. 

Libraries used are: 
io, sys, csv, math

numpy, scipy, urllib, cv2, skimage, PIL.

(I used Conda to have consistant package paths to python3).
