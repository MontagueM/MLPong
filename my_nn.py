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
        self.neurons = []


class Neuron:
    def __init__(self):
        self.incoming_neurons = []
        self.value = 0.0


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
        network.neurons.append(Neuron())

    for out in range(len(outputs)):
        network.neurons.append(Neuron())

    genome.genes = sorted(genome.genes, key=lambda x: x.out)[::-1]

    # TODO change this
    for gene in genome.genes:
        if gene.enabled:
            if gene.out not in network.neurons.keys():
                network.neurons[gene.out] = Neuron()
            neuron = network.neurons[gene.out]
            neuron.incoming.append(gene)
            if gene.into not in network.neurons.keys():
                network.neurons[gene.into] = Neuron()


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
    for genome in pool.genomes:
        if len(pool.genomes) > POPULATION:
            break

        dupe_count = genome.fitness
        while dupe_count > 0:
            if np.random.random() < BREED_PROBABILITY:
                # Breed
                child = breed_child(genome)
            else:
                # Copy
                child = copy.copy(genome)
            # We should randomly mutate some of them to incur new changes to the system
            if np.random.random() < MUTATE_PROBABILITY:
                child = mutate(child)
            pool.genomes.append(child)
            dupe_count -= 1

    pool.generation += 1


def breed_child(genome):
    pass


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
