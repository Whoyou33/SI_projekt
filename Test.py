import matplotlib.pyplot as plt
from PIL import ImageGrab
import matplotlib.image as mpimg
import numpy as np
import pyautogui


#img = ImageGrab.grab(bbox=(1311,394,1343,426))
#img_np = np.array(img)
#plt.imshow(img, cmap ="gray")
#plt.show
screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
print(pyautogui.size())
currentMouseX, currentMouseY = pyautogui.position() # Get the XY position of the mouse.
print(pyautogui.position())
currentMouseX, currentMouseY = pyautogui.position()
pyautogui.moveTo(1355,382)
pyautogui.click()
#pyautogui.moveTo(1320, 404)
#pyautogui.drag(-34, -198, 2, button='left')
#pyautogui.moveTo(1215, 404)
##plecak (1286,206)
#pyautogui.drag(4, -198, 1, button = 'left')
#pyautogui.press('enter')