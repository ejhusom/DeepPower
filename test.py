"""

counter: Hvor mange decisekunder det er siden siste topp, i.e. antall ds mellom
toppene.

counter = 10 ->  1 Hz
counter =  5 ->  2 Hz
counter =  2 ->  5 Hz

f = 10 / counter =>

"""
import matplotlib.pyplot as plt
import numpy as np

peaks = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

freq = []

counter = 0

for i, p in enumerate(peaks):

    if p == 1:
        if counter == 0:
            f = 10
        else:
            f = 10 / counter
        counter = 0
    else:
        counter += 1

    freq.append(f)


print(freq)

plt.plot(freq)
plt.show()
