"""
This may not be NEAT, and instead just an evolution system that is inspired from things I've read and seen.
Ideally much simpler to understand.
"""
import numpy as np
import pong_pygame
import copy

inputs = ['proj_x', 'proj_y', 'p0_x', 'p0_y', 'p1_x', 'p2_y']
outputs = ['Up', 'Down']

POPULATION = 300
BREED_PROBABILITY = 0.75
MUTATE_PROBABILITY = 0.2

def sigmoid(x):
    return 2/(1+np.exp(-4.9*x))-1


class Pool:
    def __init__(self):
        # With this system of genomes I no longer need to use weird randomness stuff
        self.genomes = []
        self.generation = 0
        # The current genome/species is in relation to the original arrays
        self.first_genome_index = 0

        self.max_fitness = 0
        self.current_frame = 0


class Genome:
    def __init__(self):
        self.fitness = 0
        self.network = None


class Network:
    def __init__(self):
        # We can identify inputs and outputs based on the number of inputs and outputs in the system
        self.units = []
        self.max_h_neurons = 1
        self.max_connections = 2


class Connection:
    def __init__(self):
        self.starting_unit = None
        self.weight = 0.0
        self.ending_unit = None


class Unit:
    def __init__(self):
        self.incoming_connections = []
        self.value = 0.0
        # Placeholder enabled until we may need it (with biases)
        self.enabled = True


def initialise_pool():
    """
    Fills a new pool with a set of genomes, where those genomes are filled randomly with different networks at random.
    """
    global pool
    pool = Pool()
    for i in range(POPULATION):
        genome = basic_genome()
        pool.genomes.append(genome)
    return pool


def basic_genome():
    """
    Generates a new genome with a random network of units (not fully connected).
    """
    genome = Genome()
    genome.network = generate_network(genome)
    return genome


def get_inputs():
    # pong_game.capture_screen()
    # first set of inputs will just be the projectile location and both player locations
    proj_loc = pong_game.projectile.rect.center
    player0_loc = pong_game.player0.rect.center
    player1_loc = pong_game.player1.rect.center
    return np.array([proj_loc, player0_loc, player1_loc]).flatten()


def generate_network(genome):
    network = Network()

    for inp in range(len(inputs)):
        network.units.append(Unit())

    for out in range(len(outputs)):
        network.units.append(Unit())

    # Hidden units
    num_h_units = int(np.ceil(np.random.random()*genome.network.max_h_neurons))
    num_connections = int(np.ceil(np.random.random()*genome.network.max_connections))
    for i in range(num_h_units):
        network.units.append(Unit())

    while num_connections > 0:
        conn = Connection
        b = np.sqrt(6) / np.sqrt(num_h_units)
        conn.weight = np.random.uniform(-b, b) # sample from that thing before
        # We don't want to start a connection from an output node
        conn.starting_unit = network.units[np.random.randint(0, len(network.units)-len(outputs))]
        # We don't want to end a connection from an input node
        ending_index = np.random.randint(len(inputs), len(network.units))
        conn.ending_unit = network.units[ending_index]
        if conn.starting_unit == conn.ending_unit:
            continue
        else:
            network.units[ending_index].incoming_connections.append(conn)
            num_connections -= 1

    return network


def evaluate_network(network, inputs):
    inputs += 1

    for i in range(len(inputs)):
        network.neurons[i].value = inputs[i]
    # TODO change this
    for _, neuron in network.neurons.items():
        w_sum = 0
        for incoming in neuron.incoming:
            other = network.neurons[incoming.into]
            w_sum += incoming.weight * other.value

        if len(neuron.incoming) > 0:
            neuron.value = sigmoid(w_sum)

    button_outputs = {'Up': False, 'Down': False}
    for o in range(len(outputs), 0, -1):
        if network.neurons[-o].value > 0:
            button_outputs[outputs[o]] = True
        else:
            button_outputs[outputs[o]] = False

    return button_outputs


def evaluate_current_genome(genome):
    inputs = get_inputs()
    controller = evaluate_network(genome.network, inputs)

    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

    return controller


def new_generation():
    # Reducing genomes for later breeding/copying
    pool.genomes = sorted(pool.genomes, key=lambda x: x.fitness)[::-1]
    cutoff = len(pool.genomes) * 0.1  # Taking 10% of top genomes
    pool.genomes = pool.genomes[:int(cutoff)]

    # Populating
    for i, genome in enumerate(pool.genomes):
        if len(pool.genomes) > POPULATION:
            break

        dupe_count = genome.fitness
        while dupe_count > 0:
            child = breed_child(i)
            child = mutate(child)
            pool.genomes.append(child)
            dupe_count -= 1

    pool.generation += 1


def breed_child(genome_index):
    """
    Some of the new genomes should be bred to introduce new parameters that may be more optimal.
    Breeding occurs (currently) with the genome just below itself.
    Breeding uses a normal gaussian dist ratio to interpolate the weights (and biases?) for the new child
    network.
    """
    if np.random.random() < BREED_PROBABILITY:
        # Breed
        child = copy.copy(genome)

        network1 = child.network
        network2 = pool.genomes[genome_index-1].network

        interp_ratio = np.random.normal(0.5, 1)

        new_len = int(interp_ratio * (network1[len(inputs):-len(outputs)] + network2[len(inputs):-len(outputs)])/2)
        if len(network1) > len(network2):
            child.network = network1[len(inputs)] + network1[new_len] + network1[-len(outputs)]
        else:
            child.network = network2[len(inputs)] + network2[new_len] + network2[-len(outputs)]
    else:
        # Copy
        child = copy.copy(genome)
    return child


def mutate(child):
    """
    We should randomly mutate some of the new genomes to incur new changes to the system.
    Mutation works by potentially adding the number of possible connections/neurons/etc.
    """
    for attr, value in child.network.__dict__.items():
        if attr == 'neurons':
            continue
        if np.random.random() < MUTATE_PROBABILITY:
            setattr(child.network_specs, attr, value + 1)

    return child


def initialise_run():
    # Init new game
    global pong_game
    pong_game = start_pong_game()
    pool.current_frame = 0

    # Resetting buttons
    for i in range(2):
        pong_game.press_buttons({'Up': False, 'Down': False}, genome_index=i, b_network=True)

    if pool.first_genome_index >= len(pool.genomes):
        new_generation()

    for i in range(pool.first_genome_index, pool.first_genome_index+2):
        pool.genomes[i].network = generate_network(pool.genomes[i])


def process_run():
    for i in range(pool.first_genome_index, pool.first_genome_index+2):
        genome = pool.genomes[i]
        button = evaluate_current_genome(genome)
        pong_game.press_buttons(button, genome_index=i, b_network=True)


def start_pong_game():
    # We want a single frame update so there can be some input to the network
    pong = pong_pygame.Pong()
    pong.frame()
    pong.update_frame()
    return pong


def update_pong_game():
    pool.current_frame = pool.current_frame + 1
    pong_game.frame()
    pong_game.update_frame()


if __name__ == '__main__':
    pong_game = None
    pool = initialise_pool()

    while True:
        if not pong_game:
            print(f'Gen {pool.generation}')
            initialise_run()
            process_run()

        elif pong_game.is_completed:
            for i in range(pool.first_genome_index, pool.first_genome_index+2):
                genome = pool.genomes[i]

                genome.fitness = np.exp(pool.current_frame / 50)
                if genome.fitness > pool.max_fitness:
                    pool.max_fitness = genome.fitness
                genome.genomes[genome].fitness = genome.fitness

            pool.first_genome_index += 2
            print(f'Gen {pool.generation} | Genome first index {pool.first_genome_index}')
            # TODO Shuffle genomes somewhere
            initialise_run()
            process_run()

        for i in range(pool.first_genome_index, pool.first_genome_index + 2):
            genome = pool.genomes[i]

            if pool.current_frame % 5 == 0 and pool.current_frame != 0:
                buttons = evaluate_current_genome(genome)
                pong_game.press_buttons(buttons, genome_index=i, b_network=True)

        update_pong_game()
