import copy, time
import os
import sys
import argparse
import traceback
import numpy as np
import cv2
from input_reader import InputReader
from tracker import Tracker

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-f", "--fps", type=int, help="fps", default=30)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=50)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, gaze tracking is enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(4)

input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, args.fps)

try:
    failures = 0
    source_name = input_reader.name
    ret, frame = input_reader.read()
    first = False
    height, width, channels = frame.shape
    tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, \
        max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, \
        silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, \
        no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, \
        detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, \
        max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, \
        try_hard=args.try_hard == 1)
    faces, duration = tracker.predict(frame)
    for face_num, f in enumerate(faces):
        f = copy.copy(f)
        f.id += args.face_id_offset
        frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
        frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        firsteyedelta = np.linalg.norm(f.lms[36][:2]-f.lms[45][:2])
        print(firsteyedelta)
except:
    print("ded")
    exit

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, WindowProperties
from direct.task import Task, TaskManagerGlobal

def tracking(mainapp, task):
    if input_reader.is_open():
        ret, frame = input_reader.read()
        try:
            faces, duration = tracker.predict(frame)
            origin_offset = (0,0,0)
            orientation_right=90
            radius = 20
            for face_num, f in enumerate(faces):
                f = copy.copy(f)
                f.id += args.face_id_offset
                frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
                frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                for pt_num, (x,y,c) in enumerate(f.lms):
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    if args.visualize != 0:
                        if args.visualize > 3:
                            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                        color = (255, 255, 255)
                        if pt_num >= 66:
                            color = (255, 255, 0)
                        if pt_num in [45, 36, 27]:
                            #42 leye near bridge, 45 leye outer
                            #39 reye near bridge, 36 reye outer
                            #27 bridge
                            color = (255, 0, 0)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            cv2.circle(frame, (y, x), 1, color, -1)
                x, y = f.lms[27][:2]
                eyedelta = np.linalg.norm(f.lms[36][:2]-f.lms[45][:2])

                forback = eyedelta/10
                leftright = (312.5 -y)/40

                mainapp.cam.setPos((20+eyedelta/10, (312.5 - y)/40 +20,(500-x)/40))
                mainapp.cam.lookAt((0, 35,5))
                mainapp.cam.node().getLens().setFov(eyedelta/10+100)
                # mainapp.camera.lookAt((20, 30,5))
                # print(f.lms[36][:2])
                print(eyedelta)
                print("x:",20+eyedelta/10)      #25-37
                print("y:",(312.5 - y)/40 +20)  #15-25
            if args.visualize != 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                exit
            traceback.print_exc()
            time.sleep(1)
    return Task.again
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        props = WindowProperties()
        props.set_z_order(0)    
        props.set_size(1920, 1080)
        # props.set_undecorated(True)
        # props.set_fullscreen(True)
        props.setOrigin(0, 0)
        self.win.requestProperties(props)
        self.obj = self.loader.loadModel("space.gltf", noCache=True)
        self.obj.reparentTo(self.render)
        self.obj.setScale(0.2, 0.2, 0.2)
        self.obj.setPos(-8, 42, 0)
        self.pandaActor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        self.pandaActor.loop("walk")
        self.pandaPosInterval1 = self.pandaActor.posInterval(13, Point3(0, -10, 0), startPos=Point3(0, 10, 0))
        self.pandaPosInterval2 = self.pandaActor.posInterval(13, Point3(0, 10, 0), startPos=Point3(0, -10, 0))
        self.pandaHprInterval1 = self.pandaActor.hprInterval(3, Point3(180, 0, 0), startHpr=Point3(0, 0, 0))
        self.pandaHprInterval2 = self.pandaActor.hprInterval(3, Point3(0, 0, 0), startHpr=Point3(180, 0, 0))
        self.pandaPace = Sequence(self.pandaPosInterval1, self.pandaHprInterval1, self.pandaPosInterval2, self.pandaHprInterval2,name="pandaPace")
        self.pandaPace.loop()
        self.disableMouse()
        base.camera.setPos(0, -19, 2)
        # base.camNode.setHpr(0, -10, 0)
        self.accept("arrow_right", self.mov, ["right"])
        self.accept("arrow_right-repeat", self.mov, ["right"])
        self.accept("arrow_left", self.mov, ["left"])
        self.accept("arrow_left-repeat", self.mov, ["left"])
        self.accept("arrow_down", self.mov, ["back"])
        self.accept("arrow_down-repeat", self.mov, ["back"])
        self.accept("arrow_up", self.mov, ["fwd"])
        self.accept("arrow_up-repeat", self.mov, ["fwd"])
    def mov(self, dir):
        pos = base.camera.getPos()
        # hpr = base.camera.getHpr()
        if dir == "fwd":
            base.camera.setPos(pos + (0, 1, 0))
        elif dir == "back":
            base.camera.setPos(pos - (0, 1, 0))
        elif dir == "right":
            base.camera.setPos(pos + (1, 0, 0))
        elif dir == "left":
            base.camera.setPos(pos - (1, 0, 0))


app = MyApp()
TaskManagerGlobal.taskMgr.add(tracking, appendTask=True, extraArgs=[app])
app.run()
