import sys,cv2
p,i=sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 0
cap=cv2.VideoCapture(p)

_,bg=cap.read()
cap.release()
h,w=bg.shape[:2]
print(f"Background shape: {bg.shape}")

cam=cv2.VideoCapture(i)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret,f=cam.read()
    if not ret: break
    f=f.resize((640,480,3))
    y,x=(f.shape[0]-h)//2,(f.shape[1]-w)//2
    cv2.imshow('align',cv2.addWeighted(bg,.5,f[y:y+h,x:x+w],.5,0))
    if cv2.waitKey(1)&0xFF==ord('q'): break

cam.release()
cv2.destroyAllWindows()   