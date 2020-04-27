import sys
import time

# Holds utilities that many parts of the peak tracker use.

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r\r" % (prefix, "#"*x, "."*(size-x), j, count))
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)