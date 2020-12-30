import nengo
import nengo.spa as spa
import numpy as np

import grid

mymap = """
#######
#  M  #
# # #B#
# # # #
#G Y R#
#######
"""


class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'

        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True

        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5


world = grid.World(Cell, map=mymap, directions=4)

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)


def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)


# Your model might not be a nengo.Netowrk() - SPA is permitted
model = spa.SPA()
with model:
    env = grid.GridNode(world, dt=0.005)

    movement = nengo.Node(move, size_in=2)

    # Three sensors for distance to the walls
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]


    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)

    # a basic movement function that just avoids walls based
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn


    # the movement function is only driven by information from the
    # radar
    nengo.Connection(radar, movement, function=movement_func)

    # if you wanted to know the position in the world, this is how to do it
    # The first two dimensions are X,Y coordinates, the third is the orientation
    # (plotting XY value shows the first two dimensions)
    def position_func(t):
        return body.x / world.width * 2 - 1, 1 - body.y / world.height * 2, body.dir / world.directions


    position = nengo.Node(position_func)

    # This node returns the colour of the cell currently occupied. Note that you might want to transform this into
    # something else (see the assignment)
    current_color = nengo.Node(lambda t: body.cell.cellcolor)

    D = 32
    D2 = 2
    vocab = spa.Vocabulary(D)
    vocab.parse("Green+Red+Blue+White+Magenta+Yellow")

    vocab2 = spa.Vocabulary(D2)
    vocab2.parse("Y+N")

    model.green = spa.State(D2, vocab=vocab2)
    model.red = spa.State(D2, vocab=vocab2)
    model.yellow = spa.State(D2, vocab=vocab2)
    model.magenta = spa.State(D2, vocab=vocab2)
    model.blue = spa.State(D2, vocab=vocab2)

    def convert(x):
        if x == 1:
            return vocab['Green'].v.reshape(D)
        elif x == 2:
            return vocab['Red'].v.reshape(D)
        elif x == 3:
            return vocab['Blue'].v.reshape(D)
        elif x == 4:
            return vocab['Magenta'].v.reshape(D)
        elif x == 5:
            return vocab['Yellow'].v.reshape(D)
        else:
            return vocab['White'].v.reshape(D)


    model.clean_green = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.green.output, model.clean_green.input, synapse=0.01)
    nengo.Connection(model.clean_green.output, model.green.output, synapse=0.01)

    model.clean_blue = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.blue.output, model.clean_blue.input, synapse=0.01)
    nengo.Connection(model.clean_blue.output, model.blue.output, synapse=0.01)

    model.clean_red = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.red.output, model.clean_red.input, synapse=0.01)
    nengo.Connection(model.clean_red.output, model.red.output, synapse=0.01)

    model.clean_magenta = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.magenta.output, model.clean_magenta.input, synapse=0.01)
    nengo.Connection(model.clean_magenta.output, model.magenta.output, synapse=0.01)

    model.clean_yellow = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.yellow.output, model.clean_yellow.input, synapse=0.01)
    nengo.Connection(model.clean_yellow.output, model.yellow.output, synapse=0.01)

    model.converter = spa.State(D, vocab=vocab)
    nengo.Connection(current_color, model.converter.input, function=convert)

    actions = spa.Actions(
        'dot(converter, Green) --> green=Y',
        'dot(converter, Red) --> red=Y',
        'dot(converter, Blue) --> blue=Y',
        'dot(converter, Magenta) --> magenta=Y',
        'dot(converter, Yellow) --> yellow=Y'
    )
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)
    
    def conv_func(x):
        if x =="A":
            final = x * model.convolves.output
        else:
            final = model.convolves.output
        return final
    
    model.convolves = spa.State(D2, vocab=vocab2)
    model.start = spa.State(D2, vocab=vocab2)
    
    conv_actions = spa.Actions(
        'dot(green, Y) --> convolves=convolves * Y',
        'dot(red, Y) --> convolves=convolves * Y',
        'dot(blue, Y) --> convolves=convolves * Y',
        'dot(magenta, Y) --> convolves=convolves * Y',
        'dot(yellow, Y) --> convolves=convolves * Y',
        'dot(start, Y) --> convolves=Y',
        '0.5 -->'
        )
    model.clean_convolves = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)
    nengo.Connection(model.convolves.output, model.clean_convolves.input, synapse=0.01)
    nengo.Connection(model.clean_convolves.output, model.convolves.output, synapse=0.01)
    model.conv_bg = spa.BasalGanglia(conv_actions)
    model.conv_thalamus = spa.Thalamus(model.conv_bg)