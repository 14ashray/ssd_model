import cv2
import os

from collections import defaultdict, namedtuple
from utils import prop2abs, Size

#-------------------------------------------------------------------------------
Detection = namedtuple('Detection', ['fileid', 'confidence', 'left', 'top',
                                     'right', 'bottom'])

#-------------------------------------------------------------------------------


class PascalSummary:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.boxes = defaultdict(list)

    #---------------------------------------------------------------------------
    def add_detections(self, filename, boxes):
        fileid = os.path.basename(filename)
        fileid = ''.join(fileid.split('.')[:-1])
        img = cv2.imread(filename)
        img_size = Size(img.shape[1], img.shape[0])
        for conf, box in boxes:
            xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
            if xmin < 0:
                xmin = 0
            if xmin >= img_size.w:
                xmin = img_size.w-1
            if xmax < 0:
                xmax = 0
            if xmax >= img_size.w:
                xmax = img_size.w-1
            if ymin < 0:
                ymin = 0
            if ymin >= img_size.h:
                ymin = img_size.h-1
            if ymax < 0:
                ymax = 0
            if ymax >= img_size.h:
                ymax = img_size.h-1
            det = Detection(fileid, conf, float(xmin+1), float(ymin+1),
                            float(xmax+1), float(ymax+1))
            self.boxes[box.label].append(det)

    #---------------------------------------------------------------------------
    def write_summary(self, target_dir):
        for k, v in self.boxes.items():
            filename = target_dir+'/comp4_det_test_'+k+'.txt'
            with open(filename, 'w') as f:
                for det in v:
                    line = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n" \
                        .format(det.fileid, det.confidence, det.left, det.top,
                                det.right, det.bottom)
                    f.write(line)
