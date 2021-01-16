import math

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
    speed, rotation, run_stop = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate * run_stop)
    body.go_forward(speed * dt * max_speed * run_stop)


# Your model might not be a nengo.Netowrk() - SPA is permitted:q
model = spa.SPA()
with model:
    env = grid.GridNode(world, dt=0.005)

    movement = nengo.Node(move, size_in=3)


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
    nengo.Connection(radar, movement[:2], function=movement_func)


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
    MAX_COLOURS = 3

    color_list = ["GREEN", "RED", "YELLOW", "MAGENTA", "BLUE"]

    vocab = spa.Vocabulary(D)
    vocab.parse("+".join(color_list))
    vocab.parse("WHITE")

    vocab2 = spa.Vocabulary(D)
    vocab2.parse("YES+NO")

    for color in color_list:
        exec(f"model.{color.lower()} = spa.State(D, vocab=vocab2)")


    def convert(x):
        if x == 1:
            return vocab['GREEN'].v.reshape(D)
        elif x == 2:
            return vocab['RED'].v.reshape(D)
        elif x == 3:
            return vocab['BLUE'].v.reshape(D)
        elif x == 4:
            return vocab['MAGENTA'].v.reshape(D)
        elif x == 5:
            return vocab['YELLOW'].v.reshape(D)
        else:
            return vocab['WHITE'].v.reshape(D)


    for color in color_list:
        exec(f"model.clean_{color.lower()} = spa.AssociativeMemory(vocab2, wta_output=True, threshold=0.3)")
        exec(f"nengo.Connection(model.{color.lower()}.output, model.clean_{color.lower()}.input, synapse=0.01)")
        exec(f"nengo.Connection(model.clean_{color.lower()}.output, model.{color.lower()}.output, synapse=0.01)")

    model.converter = spa.State(D, vocab=vocab)
    nengo.Connection(current_color, model.converter.input, function=convert)

    actions = spa.Actions(
        'dot(converter, GREEN) --> green=YES',
        'dot(converter, RED) --> red=YES',
        'dot(converter, BLUE) --> blue=YES',
        'dot(converter, MAGENTA) --> magenta=YES',
        'dot(converter, YELLOW) --> yellow=YES',
        '0.5 --> '
    )
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)

    def spa_to_nengo(x):
        return [1.] if vocab["YES"].dot(x) else [0.]


    model.integrator = nengo.Ensemble(500, 1, radius=4)

    for colour in color_list:
        exec(f"model.clean_{colour.lower()}.output.output = lambda t, x:x")
        exec(f"nengo.Connection(model.clean_{colour.lower()}.output, model.integrator, function=spa_to_nengo)")

    model.stop = nengo.Ensemble(1000, 1)


    def threshold(x):
        return [0.] if math.isclose(x[0], MAX_COLOURS, rel_tol=0.2) else [1.]


    nengo.Connection(model.integrator, model.stop, function=threshold)
    nengo.Connection(model.stop, movement[2])
