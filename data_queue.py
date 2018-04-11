import Queue as q
import numpy as np
import multiprocessing as mp

#-------------------------------------------------------------------------------


class DataQueue:
    #---------------------------------------------------------------------------
    def __init__(self, img_template, label_template, maxsize):
        #-----------------------------------------------------------------------
        # Figure out the data tupes, sizes and shapes of both arrays
        #-----------------------------------------------------------------------
        self.img_dtype = img_template.dtype
        self.img_shape = img_template.shape
        self.img_bc = len(img_template.tobytes())
        self.label_dtype = label_template.dtype
        self.label_shape = label_template.shape
        self.label_bc = len(label_template.tobytes())

        #-----------------------------------------------------------------------
        # Make an array pool and queue
        #-----------------------------------------------------------------------
        self.array_pool = []
        self.array_queue = mp.Queue(maxsize)
        for i in range(maxsize):
            img_buff = mp.Array('c', self.img_bc, lock=False)
            img_arr = np.frombuffer(img_buff, dtype=self.img_dtype)
            img_arr = img_arr.reshape(self.img_shape)

            label_buff = mp.Array('c', self.label_bc, lock=False)
            label_arr = np.frombuffer(label_buff, dtype=self.label_dtype)
            label_arr = label_arr.reshape(self.label_shape)

            self.array_pool.append((img_arr, label_arr))
            self.array_queue.put(i)

        self.queue = mp.Queue(maxsize)

    #---------------------------------------------------------------------------
    def put(self, img, label, boxes, *args, **kwargs):
        #-----------------------------------------------------------------------
        # Check whether the params are consistent with the data we can store
        #-----------------------------------------------------------------------
        def check_consistency(name, arr, dtype, shape, byte_count):
            if type(arr) is not np.ndarray:
                raise ValueError(name + ' needs to be a numpy array')
            if arr.dtype != dtype:
                raise ValueError('{}\'s elements need to be of type {} but is {}'
                                 .format(name, str(dtype), str(arr.dtype)))
            if arr.shape != shape:
                raise ValueError('{}\'s shape needs to be {} but is {}'
                                 .format(name, shape, arr.shape))
            if len(arr.tobytes()) != byte_count:
                raise ValueError('{}\'s byte count needs to be {} but is {}'
                                 .format(name, byte_count, len(arr.data)))

        check_consistency('img', img, self.img_dtype, self.img_shape,
                          self.img_bc)
        check_consistency('label', label, self.label_dtype, self.label_shape,
                          self.label_bc)

        #-----------------------------------------------------------------------
        # If we can not get the slot within timeout we are actually full, not
        # empty
        #-----------------------------------------------------------------------
        try:
            arr_id = self.array_queue.get(*args, **kwargs)
        except q.Empty:
            raise q.Full()

        #-----------------------------------------------------------------------
        # Copy the arrays into the shared pool
        #-----------------------------------------------------------------------
        self.array_pool[arr_id][0][:] = img
        self.array_pool[arr_id][1][:] = label
        self.queue.put((arr_id, boxes), *args, **kwargs)

    #---------------------------------------------------------------------------
    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        arr_id = item[0]
        boxes = item[1]

        img = np.copy(self.array_pool[arr_id][0])
        label = np.copy(self.array_pool[arr_id][1])

        self.array_queue.put(arr_id)

        return img, label, boxes

    #---------------------------------------------------------------------------
    def empty(self):
        return self.queue.empty()
