#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
WIDTH = 9
HEIGHT = 4

n = 100
x = np.linspace(1,200,n)

training_loss = - np.log(x) + 10
validation_loss = - np.log(x) + 10 + x*0.01

plt.figure(figsize=(WIDTH,HEIGHT))

plt.plot(training_loss, label="Training")
plt.plot(validation_loss, label="Validation")

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.xlabel("Number of epochs")
plt.ylabel("Error")
plt.legend()
plt.savefig("assets/plots/trainingloss_example.pdf")
plt.show()
