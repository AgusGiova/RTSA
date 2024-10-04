import logging
import threading
import time

def thread_function(name):
    x=0
    while x<5:
        logging.info("Thread %s: starting", name)
        time.sleep(2)
        x = x + 1

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    threads = list()
    index = 0
    logging.info("Main    : create and start thread %d.", index)
    x = threading.Thread(target=thread_function, args=(index,))
    x.start()
    
    index = index + 1
