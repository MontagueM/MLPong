"""
This may not be NEAT, and instead just an evolution system that is inspired from things I've read and seen.
Ideally much simpler to understand.
"""
import numpy as np

inputs = ['proj_x', 'proj_y', 'p0_x', 'p0_y', 'p1_x', 'p2_y']
outputs = ['Up', 'Down']

POPULATION = 300


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


def update_game():
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

            print(f'Gen {pool.generation}')
            pool.first_genome_index += 2
            # TODO Shuffle genomes somewhere
            initialise_run()

        for i in range(pool.first_genome_index, pool.first_genome_index + 2):
            genome = pool.genomes[i]

            if pool.current_frame % 5 == 0 and pool.current_frame != 0:
                buttons = evaluate_genome_network(genome)
                pong_game.press_buttons(buttons, genome_index=i, b_network=True)

        update_game()
