import logging
from itertools import islice
from multiprocessing import Manager, Process
try:
    import queue
except ImportError:
    import Queue as queue
import time
import traceback

"""
The MultiRunner framework to code multi-processor program
the client only override process_data function which get
the sliced data piece(list or dict), extra parameter for the function
is sent through kwargs as a class level parameters
"""


class MultiRunnerBase(object):
    def __init__(self, data, worker_num, timeout, **kwargs):
        self.data = data
        self.mp = MultiProcessor(worker_num=worker_num, timeout=timeout)
        self.params = kwargs
        self.data_piece = None
        self.worker_num = worker_num

    def start_runner(self, chunk_num=None):
        # slice the data by default is the number of workers
        if chunk_num is None:
            chunk_num = self.worker_num
        if isinstance(self.data, list):
            self.data_piece = MultiProcessor.chunk_list(self.data, chunk_num=chunk_num)
        elif isinstance(self.data, dict):
            self.data_piece = MultiProcessor.chunk_dict(self.data, chunk_num=chunk_num)
        else:
            logging.error('data format is wrong')
            return
        # add the data piece in the queue
        for piece in self.data_piece:
            self.mp.add_task_queue(piece)
        # start the single run process
        self.mp.start_multi_proc_with_runner(self.single_run)

    # the skeleton of single runner don't override
    def single_run(self, worker_id, task_queue, result_queue):

        logging.info('worker %d running....' % worker_id)
        self.params['worker_id'] = worker_id
        while True:
            try:
                if not task_queue.empty():
                    data = task_queue.get(False)
                    ret = self.process_data(data)
                    result_queue.put(ret)
                else:
                    time.sleep(1)
                    if task_queue.empty():
                        break
                    continue
            except Exception as err:
                logging.info(err)
                print(traceback.print_exc())
                break

    # the function for override, extra params are in self.params the return result is put
    # into the result queue
    def process_data(self, data):
        print(self.params)
        return data

    # get the result with iterator
    def iter_result(self):
        while self.mp.get_result_queue_size() > 0:
            yield self.mp.pop_result_queue()


class MultiProcessor(object):

    def __init__(self, worker_num=20, timeout=60):
        # setup the queue
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.worker_num = worker_num
        self.timeout = timeout

    def start_multi_proc(self, run_logic, **kwargs):
        # allocate the processors
        process_list = []
        for worker_id in range(self.worker_num):
            p = Process(target=run_logic, args=(worker_id, self.task_queue, self.result_queue, kwargs))
            process_list.append(p)
        logging.info('%d worker initialized....' % len(process_list))
        logging.info('start running in multiple process mode.....')
        # start the processors
        for p in process_list:
            p.start()
        logging.info('allocate the result.....')
        # wait all finished with a timeout
        for p in process_list:
            p.join(timeout=self.timeout)

    def start_multi_proc_with_runner(self, run_logic):

        process_list = []
        for worker_id in range(self.worker_num):
            p = Process(target=run_logic, args=(worker_id, self.task_queue, self.result_queue))
            process_list.append(p)

        print('%d worker initialized....' % len(process_list))
        print('start running in multiple process mode.....')
        for p in process_list:
            p.start()
        print('allocate the result.....')
        for p in process_list:
            p.join(timeout=self.timeout)

        for p in process_list:
            p.terminate()
            p.join()

    def add_task_queue(self, data):
        self.task_queue.put(data)

    def pop_result_queue(self):
        if self.result_queue.empty():
            return None
        else:
            # if have exception, try once more
            cn = 0
            while cn < 2:
                try:
                    return self.result_queue.get(False)
                except Queue.Empty as e:
                    logging.error('queue empty exception, retry')
                    logging.info(e)
                    cn += 1
                    time.sleep(1)
            return None

    def reset_processor(self):
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()

    @staticmethod
    def chunk_dict(data, chunk_num):
        chunk_size = int(len(data.keys()) / chunk_num)
        if chunk_size == 0:
            chunk_size = 1
        it = iter(data)
        for i in range(0, len(data), chunk_size):
            yield {k: data[k] for k in islice(it, chunk_size)}

    @staticmethod
    def chunk_list(data, chunk_num):
        chunk_size = int(len(data) / chunk_num)
        n = max(1, chunk_size)
        return (data[i:i + n] for i in range(0, len(data), n))

    def get_result_queue_size(self):
        return self.result_queue.qsize()

    def get_task_queue_size(self):
        return self.task_queue.qsize()


"""
Example of how to use MultiProcessor class without MultiRunner Framework
the client have to handle all queue operations himself, however it is more 
flexible
"""


def example():

    mp = MultiProcessor(worker_num=2, timeout=100)
    source_list = [i for i in range(10)]
    # slice the data into the same piece of worker
    slices = MultiProcessor.chunk_list(source_list, chunk_num=2)
    # add each piece into task queue
    for s in slices:
        mp.add_task_queue(s)
    # start with self-defined single run function
    mp.start_multi_proc(single_run_example, n=2)
    # read the result from the queue
    while mp.get_result_queue_size() > 0:
        ret = mp.pop_result_queue()
        if ret:
            print(ret)


def single_run_example(worker_id, task_queue, result_queue, args):
    # logging the worker id
    logging.info('worker %d running....' % worker_id)
    # read from the task queue, process and write to result queue
    n = args['n'] if 'n' in args else 1
    while True:
        try:
            if not task_queue.empty():
                # read from the queue
                ele = task_queue.get(False)
                # process the element and add into queue
                result_queue.put([e*n for e in ele])
            else:
                time.sleep(1)
                if task_queue.empty():
                    break
                continue
        except Exception as err:
            logging.info(err)
            logging.info(traceback.print_exc())
            break


"""
the example of how to use MultiRunner skeleton to run multi-process program
"""


class MultiRunnerExample(MultiRunnerBase):

    # not init large variable here, the function will be called
    # everytime the processor fetch a item from the task queue
    # put large var in params while call the class and refer by params['x']
    def process_data(self, data):
        return [d*self.params['n'] for d in data]


if __name__ == '__main__':

    example()

    mre = MultiRunnerExample(data=[x for x in range(10)], worker_num=2,
                             timeout=100, n=4)
    mre.start_runner()
    for r in mre.iter_result():
        print(r)

