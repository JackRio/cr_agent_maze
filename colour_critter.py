import math

import nengo
import nengo.spa as spa
import numpy as np

import grid

MAX_COLOURS = 3  # change if you want to detect more or less colours

mymap = """
#######
#  M  #
# # # #
# #B# #
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
    """
    This function calculates the rotation speed and moving speed of the agent.
    :param x: State Input
    :type x: float
    """
    speed, rotation, run_stop = x
    dt = 0.001
    max_speed = 10.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate * run_stop)
    body.go_forward(speed * dt * max_speed * run_stop)


# Your model might not be a nengo.Netowrk() - SPA is permitted:q
model = spa.SPA()
with model:

    env = grid.GridNode(world, dt=0.005)
    movement = nengo.Node(move, size_in=3)

    def detect(t):
        """
        This function calculates the distance of the agent from the walls in all three direction
        left, right and forward
        :return: sensory information about the distance from walls
        :rtype: list
        """
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]

    def movement_func(x):
        # x[0] = senosor in the left --> np "first black square to the critter
        # x[1] = sensory in the front.
        # the closer the wall is the slower it goes.
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn

    stim_radar = nengo.Node(detect)
    radar = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)

    nengo.Connection(stim_radar, radar)
    nengo.Connection(radar, movement[:2], function=movement_func)

    # This node returns the colour of the cell currently occupied. Note that you might want to transform this into
    # something else (see the assignment)
    current_color = nengo.Node(lambda t: body.cell.cellcolor)

    # Variables used in the code
    D = 32
    n_neurons = 1000

    # The list of colours available in the environment/maze
    color_list = ["GREEN", "RED", "YELLOW", "MAGENTA", "BLUE"]
    colour_vocab = spa.Vocabulary(D)
    colour_vocab.parse("+".join(color_list))
    colour_vocab.parse("WHITE")

    colour_state_vocab = spa.Vocabulary(D)
    colour_state_vocab.parse("YES+NO")  # this is what freddy helped us with

    # make the list of colours / adding all the colour states
    for color in color_list:
        exec(f"model.{color.lower()} = spa.State(D, vocab=colour_state_vocab)")


    # the colour detection. convert numbers into a spa vector (?)
    def convert(x):
        """
        This function converts the integral value into the corresponding semantic pointer representing the colour
        :param x: State input
        :type x: float
        :return: Semantic pointer
        :rtype: Vector
        """
        if x == 1:
            return colour_vocab['GREEN'].v.reshape(D)
        elif x == 2:
            return colour_vocab['RED'].v.reshape(D)
        elif x == 3:
            return colour_vocab['BLUE'].v.reshape(D)
        elif x == 4:
            return colour_vocab['MAGENTA'].v.reshape(D)
        elif x == 5:
            return colour_vocab['YELLOW'].v.reshape(D)
        else:
            return colour_vocab['WHITE'].v.reshape(D)


    # model.clean = memory clean up to stabalize
    # and all the connections
    for color in color_list:
        exec(f"model.clean_{color.lower()} = spa.AssociativeMemory(colour_state_vocab, wta_output=True, threshold=0.3)")
        exec(f"nengo.Connection(model.{color.lower()}.output, model.clean_{color.lower()}.input, synapse=0.01)")
        exec(f"nengo.Connection(model.clean_{color.lower()}.output, model.{color.lower()}.output, synapse=0.01)")

    model.converter = spa.State(D, vocab=colour_vocab)
    nengo.Connection(current_color, model.converter.input, function=convert)

    # if a colour is detected, then output YES for that colour
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
        """
        Converting spa input into float value for the nengo Ensemble
        :param x: State Input
        :type x: float
        :return: 1 or 0 based on if the colour state is activated
        :rtype: list
        """
        return [1.] if colour_vocab["YES"].dot(x) else [0.]

    # Ensemble that counts the number of coloured tiles agent has seen.
    model.counter = nengo.Ensemble(1000, 1, radius=5)

    # Thijs Gelton helped us with the implementation of this part.
    for colour in color_list:
        exec(f"model.clean_{colour.lower()}.output.output = lambda t, x:x")
        exec(f"nengo.Connection(model.clean_{colour.lower()}.output, model.counter, function=spa_to_nengo)")

    model.stop = nengo.Ensemble(n_neurons, 1)
    inhib = nengo.Node(size_in=1)


    def inhibit(x):
        """
        This function triggers the inhib node which in turns set the value
        of the stop neurons to exact 0 (i.e They are inhibited)
        """
        if math.isclose(x[0], MAX_COLOURS):
            return [1.]
        elif x[0] > MAX_COLOURS:
            # This is just in case we have colours in sequential pattern
            return [1.]
        else:
            return [0.]


    # Provides input to the inhibit node to stop neuron activity when the agent has crossed the threshold
    nengo.Connection(model.counter, inhib, function=inhibit)

    # Stops the neuron activity
    nengo.Connection(
        inhib, model.stop.neurons,
        transform=-10 * np.ones((n_neurons, 1))
    )
    nengo.Connection(model.counter, model.stop, function=lambda x: x / x)
    # Updating the movement function to make the agent move or stop at its track
    nengo.Connection(model.stop, movement[2])
