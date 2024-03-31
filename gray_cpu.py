import multiprocessing
import signal

import numpy as np

def get_cpu_count():
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 1
    return cpu_count


def get_segments(image_src):
    cpu_count = get_cpu_count()
    per_process_x = image_src.shape[1] // cpu_count
    per_process_y = image_src.shape[0] // cpu_count

    segments = []
    for i in range(cpu_count):
        start_y = i * per_process_y
        if i == (cpu_count - 1):
            end_y = image_src.shape[0]
        else:
            end_y = start_y + per_process_y

        for j in range(cpu_count):
            start_x = j * per_process_x

            if j == (cpu_count - 1):
                end_x = image_src.shape[1]
            else:
                end_x = start_x + per_process_x

            segments.append(((start_x, end_x), (start_y, end_y)))

    return segments


def apply_cpu(image_src):
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise_sigint = False
    with multiprocessing.Pool(get_cpu_count()) as pool:
        signal.signal(signal.SIGINT, sigint_handler)
        results = []
        for s in get_segments(image_src):
            start_x, end_x = s[0]
            start_y, end_y = s[1]
            result = pool.apply_async(
                apply_filter_cpu,
                (image_src[start_y:end_y, start_x:end_x],)
            )
            results.append((result, (start_x, end_x), (start_y, end_y)))
        image_dst = np.empty_like(image_src)
        for r, (start_x, end_x), (start_y, end_y) in results:
            try:
                segment = r.get()
                image_dst[start_y:end_y, start_x:end_x] = segment
            except KeyboardInterrupt:
                pool.terminate()
                raise_sigint = True
                break
        pool.close()
        pool.join()

    if raise_sigint:
        raise KeyboardInterrupt  # catch it in filter.py

    return image_dst

def apply_filter_cpu(segment):
    image_dst = np.empty_like(segment)
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            x, y, z = segment[i, j]
            intensity = int(0.2126 * x + 0.7152 * y + 0.0722 * z)
            image_dst[i, j] = (intensity,) * 3
    return image_dst