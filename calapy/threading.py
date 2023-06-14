

__all__ = ['MultiThreads']


import os
import math
import threading
# import concurrent.futures as cf
# import time


# def run(func, args=None, kwargs=None, n_workers=None):
#
#     if args is None:
#         args = ((),)
#         n_elements_in_args = 1
#     else:
#         try:
#             n_elements_in_args = len(args)
#         except TypeError:
#             raise ValueError('args must be an iterable of an iterable')
#
#         for i in range(0, n_elements_in_args, 1):
#             if args[i] is None:
#                 args[i] = ()
#             else:
#                 try:
#                     len(args[i])
#                 except TypeError:
#                     raise ValueError('args must be an iterable of an iterable')
#
#     if kwargs is None:
#         n_elements_in_kwargs = 1
#         kwargs = ({},)
#     else:
#         try:
#             n_elements_in_kwargs = len(kwargs)
#         except TypeError:
#             raise ValueError('kwargs must be an iterable of an dicts')
#
#         for i in range(0, n_elements_in_kwargs, 1):
#             if kwargs[i] is None:
#                 kwargs[i] = {}
#             else:
#                 try:
#                     len(kwargs[i])
#                 except TypeError:
#                     raise ValueError('kwargs must be an iterable of an dicts')
#
#     if n_elements_in_args != n_elements_in_kwargs:
#         if n_elements_in_args < n_elements_in_kwargs:
#             if n_elements_in_args == 1:
#                 args = tuple([args[0] for i in range(0, n_elements_in_kwargs, 1)])
#             elif n_elements_in_args > 1:
#                 raise ValueError('args cannot broadcast to kwargs')
#             elif n_elements_in_args == 0:
#                 args = tuple([args for i in range(0, n_elements_in_kwargs, 1)])
#             n_elements_in_args = n_elements_in_kwargs
#         else:
#             if n_elements_in_kwargs == 1:
#                 kwargs = tuple([kwargs[0] for i in range(0, n_elements_in_args, 1)])
#             elif n_elements_in_kwargs > 1:
#                 raise ValueError('kwargs cannot broadcast to args')
#             elif n_elements_in_kwargs == 0:
#                 kwargs = tuple([kwargs for i in range(0, n_elements_in_args, 1)])
#             n_elements_in_kwargs = n_elements_in_args
#
#     n_processes = n_elements_in_args
#
#     # if n_workers is None:
#     #     n_cpus = os.cpu_count()
#     #     n_workers = min([n_cpus, n_processes])
#
#     processes = [None for i in range(0, n_processes, 1)]  # type: list
#     results = [None for i in range(0, n_processes, 1)]  # type: list
#
#     executor = cf.ThreadPoolExecutor(
#         max_workers=n_workers, thread_name_prefix='calapy.parallel_processes', initializer=None, initargs=())
#
#     for i in range(0, n_processes, 1):
#
#         args_i = args[i]
#         kwargs_i = kwargs[i]
#         processes[i] = executor.submit(func, *args_i, **kwargs_i)
#
#     for i in range(0, n_processes, 1):
#         results[i] = processes[i].result()
#
#     executor.shutdown(wait=True, cancel_futures=False)
#
#     return results


class MultiThreads:

    def __init__(self, func, args=None, kwargs=None, n_workers=None, names=None):

        self.func = func

        if args is None:
            self.args = ((),)
            n_elements_in_args = 1
        else:
            try:
                n_elements_in_args = len(args)
            except TypeError:
                raise ValueError('args must be an iterable of an iterable')

            self.args = args
            for i in range(0, n_elements_in_args, 1):
                if self.args[i] is None:
                    self.args[i] = ()
                else:
                    try:
                        len(self.args[i])
                    except TypeError:
                        raise ValueError('args must be an iterable of an iterable')

        if kwargs is None:
            n_elements_in_kwargs = 1
            self.kwargs = ({},)
        else:
            try:
                n_elements_in_kwargs = len(kwargs)
            except TypeError:
                raise ValueError('kwargs must be an iterable of an dicts')

            self.kwargs = kwargs
            for i in range(0, n_elements_in_kwargs, 1):
                if self.kwargs[i] is None:
                    self.kwargs[i] = {}
                else:
                    try:
                        len(self.kwargs[i])
                    except TypeError:
                        raise ValueError('kwargs must be an iterable of an dicts')

        if n_elements_in_args != n_elements_in_kwargs:
            if n_elements_in_args < n_elements_in_kwargs:
                if n_elements_in_args == 1:
                    self.args = tuple([self.args[0] for i in range(0, n_elements_in_kwargs, 1)])
                elif n_elements_in_args > 1:
                    raise ValueError('args cannot broadcast to kwargs')
                elif n_elements_in_args == 0:
                    self.args = tuple([self.args for i in range(0, n_elements_in_kwargs, 1)])
                n_elements_in_args = n_elements_in_kwargs
            else:
                if n_elements_in_kwargs == 1:
                    self.kwargs = tuple([self.kwargs[0] for i in range(0, n_elements_in_args, 1)])
                elif n_elements_in_kwargs > 1:
                    raise ValueError('kwargs cannot broadcast to args')
                elif n_elements_in_kwargs == 0:
                    self.kwargs = tuple([self.kwargs for i in range(0, n_elements_in_args, 1)])
                n_elements_in_kwargs = n_elements_in_args

        self.n_elements = n_elements_in_args
        self.results = [None for i in range(0, self.n_elements, 1)]  # type: list
        self.returned_results = [False for i in range(0, self.n_elements, 1)]  # type: list

        if n_workers is None:
            self.n_workers = min(os.cpu_count() * 2, self.n_elements)
        elif isinstance(n_workers, int):
            if n_workers > 0:
                self.n_workers = min(n_workers, self.n_elements)
            else:
                raise ValueError('n_workers must be greater than 0')

        if names is None:
            self.names = [None for j in range(0, self.n_workers, 1)]
        else:
            self.names = names

        # # todo: chuncks
        # chunk_args = args
        # chunk_kwargs = kwargs

        self.max_chunk_sizes = math.ceil(self.n_elements / self.n_workers)
        n_remainders = (self.max_chunk_sizes * self.n_workers) - self.n_elements
        if n_remainders > 0:
            self.min_chunk_sizes = self.max_chunk_sizes - 1
            index = self.n_workers - n_remainders
            self.chunk_sizes = [
                self.max_chunk_sizes if j < index else self.min_chunk_sizes for j in range(0, self.n_workers, 1)]
        else:
            self.min_chunk_sizes = self.max_chunk_sizes
            self.chunk_sizes = [self.max_chunk_sizes for j in range(0, self.n_workers, 1)]

        self.stops_elements = [0 for j in range(0, self.n_workers, 1)]
        cum_sum = 0
        for j in range(0, self.n_workers, 1):
            cum_sum += self.chunk_sizes[j]
            self.stops_elements[j] = cum_sum

        self.starts_elements = [0] + self.stops_elements[slice(0, self.n_workers - 1, 1)]

        self.workers = [threading.Thread(
            group=None, target=self.run_single_chunk, name=self.names[j], args=(), kwargs={'j': j}, daemon=None)
            for j in range(0, self.n_workers, 1)]  # type: list[threading.Thread]

        return None

    def run_single_chunk(self, j):

        for i in range(self.starts_elements[j], self.stops_elements[j], 1):

            args_i = self.args[i]
            kwargs_i = self.kwargs[i]

            self.results[i] = self.func(*args_i, **kwargs_i)
            self.returned_results[i] = True

        return None

    def start(self):

        for j in range(0, self.n_workers, 1):
            self.workers[j].start()

        return None

    def run(self):

        self.start()

        self.join()

        return self.results

    def join(self):

        # i = 0
        # while i < self.n_elements:
        #     if self.returned_results[i]:
        #         i += 1
        #     else:
        #         time.sleep(1.0)

        j = 0
        while j < self.n_workers:
            try:
                self.workers[j].join()
                j += 1
            except RuntimeError:
                pass

        return None

