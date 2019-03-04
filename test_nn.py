import numpy as np
from nn.layers import Conv2D, Dense, PReLU
from nn.optimizers import Adam
from nn.losses import softmax
from nn.model import BaseModel

batch_size = 32
nb_classes = 10
x = np.random.rand(batch_size, 3, 64, 64)
y = np.random.randint(nb_classes, size=batch_size)

class Model(BaseModel):
    def predictor(self, inp, outp):
        model = []
        model.append(Conv2D(inp, 32))
        model.append(PReLU(model[-1]))
        model.append(Dense(model[-1], 128))
        model.append(PReLU(model[-1]))
        model.append(Dense(model[-1], nb_classes))
        return model

model = Model(x, y, softmax, Adam(1e-3))
for _ in range(100):  # train 100 steps
    print(model.fit(x, y))  # loss should go down
