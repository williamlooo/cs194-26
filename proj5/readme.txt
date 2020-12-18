Hello, 

To run this code, call 'python3 main.py' in the terminal. There are two argument flags which can be included: --rectify or --manual. By default, no input will be required from the user, and both rectification and manual selection are not active.

Adding the --rectify flag will rectify image 2. Select 4 points around the frame you want to rectify, and the output (canvas.jpg) will be the rectified image.

Adding the --manual flag will run the code with manual keypoint selection. You will be asked to select 8 key points for the two images. The output (canvas.jpg) will be a stitched mosaic. By default, main.py will autoselect the points for you.

HOW TO CHANGE INPUT IMAGES: Adjust IM_1_NAME and IM_2_NAME variables on line 285/286 in main.py

REPORT: https://inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/proj5B/cs194-26-acx
