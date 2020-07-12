import numpy as np
import pong_pygame
import copy

"""
Need to rework most of this stuff to work for mutable variables instead of pointer vars
eg converting for loops into non-mutable iterators
"""

#######

inputs = ['proj_x', 'proj_y', 'p0_x', 'p0_y', 'p1_x', 'p2_y']
outputs = ['Up', 'Down']

Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

TimeoutConstant = 20

MaxNodes = 1000000


def sigmoid(x):
    return 2/(1+np.exp(-4.9*x))-1


def new_innovation():
    pool.innovation += 1
    return pool.innovation


class Pool:
    def __init__(self):
        self.species = []
        self.seen_species = []
        self.reduced_species = []
        self.generation = 0
        # Find out if innovations are required
        self.innovation = len(outputs)
        # The current genome/species is in relation to the original arrays
        self.current_species_index = [-1, -1]
        self.current_genomes_index = [-1, -1]
        self.current_frame = 0
        self.max_fitness = 0


class Species:
    def __init__(self):
        self.top_fitness = 0
        self.staleness = 0
        self.genomes = []
        self.seen_genomes = []
        self.reduced_genomes = []
        self.average_fitness = 0


class Genome:
    def __init__(self):
        self.genes = []
        self.adjusted_fitness = 0
        self.network = []
        self.max_neuron = 0
        self.global_rank = 0
        self.mutation_rates = MutationRates()


class Gene:
    def __init__(self):
        self.into = 0
        self.out = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0


class MutationRates:
    def __init__(self):
        self.connections = MutateConnectionsChance
        self.link = LinkMutationChance
        self.bias = BiasMutationChance
        self.node = NodeMutationChance
        self.enable = EnableMutationChance
        self.disable = DisableMutationChance
        self.step = StepSize


class Neuron:
    def __init__(self):
        self.incoming = []
        self.value = 0.0


class Network:
    def __init__(self):
        self.neurons = []


def generate_network(genome):
    network = Network()
    network.neurons = {}

    for inp in range(len(inputs)):
        network.neurons[inp] = Neuron()

    for out in range(len(outputs)):
        network.neurons[MaxNodes+out] = Neuron()

    genome.genes = sorted(genome.genes, key=lambda x: x.out)

    for gene in genome.genes:
        if gene.enabled:
            if gene.out not in network.neurons.keys():
                network.neurons[gene.out] = Neuron()
            neuron = network.neurons[gene.out]
            neuron.incoming.append(gene)
            if gene.into not in network.neurons.keys():
                network.neurons[gene.into] = Neuron()

    return network


def rank_globally():
    glob = []
    for species in pool.species:
        for g in species.genomes:
            glob.append(g)

    glob = sorted(glob, key=lambda x: x.fitness)
    glob = glob[::-1]
    # TODO make sure this fits global rank as highest fitness (ie correct sorting)
    for i in range(len(glob)):
        # TODO make sure this actually sets it (pointer vs reference) mutable
        glob[i].global_rank = i
    print('Max fitness:', glob[0].fitness)


def calculate_average_fitness(species):
    total = 0
    # TODO check this total works mutable
    for gen in species.genomes:
        total += gen.global_rank

    species.average_fitness = total / len(species.genomes)


def total_average_fitness():
    total = 0
    # TODO check this total works mutable
    for species in pool.species:
        total += species.average_fitness

    return total


def remove_stale_species():
    survived = []
    for species in pool.species:
        species.genomes = sorted(species.genomes, key=lambda x: x.fitness)

        if species.genomes[0].fitness > species.top_fitness:
            species.top_fitness = species.genomes[0].fitness
            species.staleness = 0
        else:
            species.staleness += 1

        if species.staleness < StaleSpecies or species.top_fitness >= pool.max_fitness:
            survived.append(species)

    pool.species = survived


def remove_weak_species():
    survived = []

    sum = total_average_fitness()
    for species in pool.species:
        breed = np.floor(species.average_fitness / sum * Population)
        if breed >= 1:
            survived.append(species)
    pool.species = survived


def cull_species(cull_to_one):
    for i, sp in enumerate(pool.species):
        sp.genomes = sorted(sp.genomes, key=lambda x: x.fitness)

        remaining = int(np.ceil(len(sp.genomes)/2))

        if cull_to_one:
            remaining = 1

        pool.species[i].genomes = sp.genomes[-remaining:]


def evaluate_network(network, inputs):
    inputs += 1
    # TODO check this line
    # if inputs != len(inputs):
    #     print("Incorrect number of neural network inputs.")
    #     return
    # print("Q")
    for i in range(len(inputs)):
        network.neurons[i].value = inputs[i]

    for _, neuron in network.neurons.items():
        sum_ = 0
        for incoming in neuron.incoming:
            other = network.neurons[incoming.into]
            sum_ += incoming.weight * other.value

        if len(neuron.incoming) > 0:
            neuron.value = sigmoid(sum_)

    button_outputs = {'Up': False, 'Down': False}
    for o in range(len(outputs)):
        if network.neurons[MaxNodes+o].value > 0:
            button_outputs[outputs[o]] = True
        else:
            button_outputs[outputs[o]] = False

    return button_outputs


def breed_child(species):
    if np.random.random() < CrossoverChance:
        g1 = species.genomes[np.random.randint(len(species.genomes))]
        g2 = species.genomes[np.random.randint(len(species.genomes))]
        child = crossover(g1, g2)
    else:
        g = species.genomes[np.random.randint(len(species.genomes))]
        child = copy.copy(g)

    mutate(child)

    return child


def crossover(g1, g2):
    # Make sure g1 is the higher fitness genome
    if g2.fitness > g1.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg

    child = Genome()

    innovations2 = {}
    for gene in g2.genes:
        innovations2[gene.innovation] = gene

    for gene1 in g1.genes:
        if gene1.innovation not in innovations2.keys():
            child.genes.append(gene1)
        else:
            gene2 = innovations2[gene1.innovation]
            if gene2 != 0 and np.random.randint(2) == 1 and gene2.enabled:
                child.genes.append(gene2)
            else:
                child.genes.append(gene1)

    child.max_neuron = np.max([g1.max_neuron, g2.max_neuron])

    for mutation, rate in g1.mutation_rates.__dict__.items():
        child.mutation_rates.mutation = rate

    return child


def mutate(genome):
    print('Marker', genome)
    for mutation, rate in genome.mutation_rates.__dict__.items():
        if np.random.randint(2) == 1:
            setattr(genome.mutation_rates, mutation, 0.95*rate)
        else:
            setattr(genome.mutation_rates, mutation, 1.05263*rate)

    if np.random.random() < genome.mutation_rates.connections:
        point_mutate(genome)

    while genome.mutation_rates.link > 0:
        if np.random.random() < genome.mutation_rates.link:
            # print('a', len(genome.genes))
            ret = link_mutate(genome, False)
            if ret:
                genome = ret
                genome.mutation_rates.link -= 1
            elif pool.generation > 0:
                genome.mutation_rates.link -= 1
            # print('b', len(genome.genes))

    while genome.mutation_rates.bias > 0:
        if np.random.random() < genome.mutation_rates.bias:
            ret = link_mutate(genome, True)
            if ret:
                genome = ret

        genome.mutation_rates.bias -= 1

    while genome.mutation_rates.node > 0:
        if np.random.random() < genome.mutation_rates.node:
            ret = node_mutate(genome)
            if ret:
                genome = ret

        genome.mutation_rates.node -= 1

    while genome.mutation_rates.enable > 0:
        if np.random.random() < genome.mutation_rates.enable:
            enable_disable_mutate(genome, True)

        genome.mutation_rates.enable -= 1

    while genome.mutation_rates.disable > 0:
        if np.random.random() < genome.mutation_rates.disable:
            enable_disable_mutate(genome, False)

        genome.mutation_rates.disable -= 1


def point_mutate(genome):
    step = genome.mutation_rates.step

    for i in range(len(genome.genes)):
        if np.random.random() < PerturbChance:
            genome.genes[i].weight += np.random.random() * step * 2 - step
        else:
            genome.genes[i].weight = np.random.random() * 4 - 2
    return genome


def link_mutate(genome, force_bias):
    neuron1 = random_neuron(genome.genes, False)
    neuron2 = random_neuron(genome.genes, True)

    new_link = Gene()
    if neuron1 <= len(inputs) and neuron2 <= len(inputs):
        # Both input nodes
        print("We haven't added one 1")
        return
    if neuron2 <= len(inputs):
        # Swap output and input
        temp = neuron1
        neuron1 = neuron2
        neuron2 = temp

    new_link.into = neuron1
    new_link.out = neuron2
    if force_bias:
        new_link.into = len(inputs)

    if contains_link(genome.genes, new_link):
        print("We haven't added one 2")
        return

    new_link.innovation = new_innovation()
    new_link.weight = np.random.random() * 4 - 2
    genome.genes.append(new_link)
    print('We added one')
    return genome


def random_neuron(genes, non_input):
    neurons = {}
    if not non_input:
        for i in range(len(inputs)):
            neurons[i] = True
    for o in range(len(outputs)):
        neurons[MaxNodes+o] = True

    for gene in genes:
        if not non_input or gene.into > len(inputs):
            neurons[gene.into] = True
        if not non_input or gene.out > len(inputs):
            neurons[gene.out] = True

    count = 0
    for _, _ in neurons.items():
        count += 1

    n = np.random.randint(count)

    for k, v in neurons.items():
        n -= 1
        if n == 0:
            return k
    return 0


def contains_link(genes, link):
    for gene in genes:
        if gene.into == link.into and gene.out == link.out:
            return True


def node_mutate(genome):
    if len(genome.genes) == 0:
        return

    genome.max_neuron += 1

    gene = genome.genes[np.random.randint(len(genome.genes))]

    if not gene.enabled:
        return

    gene.enabled = False

    gene1 = copy.copy(gene)
    gene1.out = genome.max_neuron
    gene1.weight = 1
    gene1.innovation = new_innovation()
    gene1.enabled = True
    genome.genes.append(gene1)

    gene2 = copy.copy(gene)
    gene2.out = genome.max_neuron
    gene2.innovation = new_innovation()
    gene2.enabled = True
    genome.genes.append(gene2)
    return genome


def enable_disable_mutate(genome, enable):
    candidates = []
    for i in range(len(genome.genes)):
        if genome.genes[i].enabled != enable:
            candidates.append(i)

    if len(candidates) == 0:
        return

    rand_index = np.random.randint(len(candidates))
    gene = genome.genes[candidates[rand_index]]
    gene.enabled = not gene.enabled
    genome.genes[rand_index] = gene


def get_inputs():
    # pong_game.capture_screen()
    # first set of inputs will just be the projectile location and both player locations
    proj_loc = pong_game.projectile.rect.center
    player0_loc = pong_game.player0.rect.center
    player1_loc = pong_game.player1.rect.center
    return np.array([proj_loc, player0_loc, player1_loc]).flatten()


def evaluate_current_genome(genome):
    inputs = get_inputs()
    # print('a', genome)
    # print('k')
    controller = evaluate_network(genome.network, inputs)

    # TODO Fix this
    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

    return controller


def initialise_run():
    global pong_game
    pong_game = start_pong_game()
    pool.current_frame = 0
    # Resetting buttons
    for i in range(len(pool.current_genomes_index)):
        pong_game.press_buttons({'Up': False, 'Down': False}, genome_index=i, b_network=True)

    reduced_species = list(set(pool.species) - set(pool.seen_species))
    pool.reduced_species = reduced_species
    if set(pool.species) == set(pool.seen_species):
        print('New generation')
        new_generation()
        pool.seen_species = []
        for sp in pool.species:
            sp.seen_genomes = []
        reduced_species = list(set(pool.species) - set(pool.seen_species))
        pool.reduced_species = reduced_species

    for i in range(2):
        _, genom = get_new_indexes(i)

        if len(genom.genes) == 0:
            print('Trying again')
            initialise_pool()
            return
            # quit()

        network = generate_network(genom)
        genom.network = network
        species = pool.species[pool.current_species_index[i]]
        species.genomes[pool.current_genomes_index[i]].network = network
        button = evaluate_current_genome(genom)
        # print(genom, genom.network)
        pong_game.press_buttons(button, genome_index=i, b_network=True)


def get_new_indexes(i):
    # current indexes are in relation to original species/genome arrays
    red_species_index = np.random.randint(len(pool.reduced_species))
    red_species = pool.reduced_species[red_species_index]
    pool.current_species_index[i] = pool.species.index(red_species)

    reduced_genomes = list(set(red_species.genomes) - set(red_species.seen_genomes))
    pool.reduced_species[red_species_index].reduced_genomes = reduced_genomes

    red_genome_index = np.random.randint(len(reduced_genomes))
    red_genome = red_species.reduced_genomes[red_genome_index]
    pool.current_genomes_index[i] = red_species.genomes.index(red_genome)

    # Making sure that we mark the species as seen if all genomes in it are seen

    pool_species = pool.species[pool.current_species_index[i]]
    # print(f'Adding seen genome {red_genome}')
    pool_species.seen_genomes.append(red_genome)
    if set(pool_species.seen_genomes) == set(pool_species.genomes):
        # print(f'Adding seen species {red_species} as {set(pool_species.seen_genomes)}, {set(pool_species.genomes)} equal')
        pool.seen_species.append(red_species)
        pool.reduced_species.remove(red_species)

    return pool_species, red_genome


def next_genomes():
    for i in range(2):
        # Selecting two random genomes from random species, making sure that we don't
        # select the same one again (trying to be efficient about it)
        reduced_species = list(set(pool.species) - set(pool.seen_species))
        pool.reduced_species = reduced_species
        pool_species, _ = get_new_indexes(i)

        if set(pool.species) == set(pool.seen_species):
            new_generation()
            next_genomes()


def new_generation():
    cull_species(cull_to_one=False)  # Cull the bottom half of each species
    remove_stale_species()
    rank_globally()
    for species in pool.species:
        calculate_average_fitness(species)
    remove_weak_species()
    sum_ = total_average_fitness()
    children = []
    for species in pool.species:
        breed = int(np.floor(species.average_fitness / sum_ * Population)) - 1
        for i in range(breed):
            children.append(breed_child(species))

    cull_species(cull_to_one=True)  # Cull all but top member of each species
    while len(children) + len(pool.species) < Population:
        species = pool.species[np.random.randint(len(pool.species))]
        children.append(breed_child(species))

    for child in children:
        add_to_species(child)

    pool.generation += 1

def add_to_species(child):
    found_species = False
    for i in range(len(pool.species)):
        if not found_species and same_species(child, pool.species[i].genomes[0]):
            pool.species[i].genomes.append(child)
            found_species = True

    if not found_species:
        child_species = Species()
        child_species.genomes.append(child)
        pool.species.append(child_species)


def same_species(genome1, genome2):
    dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
    dw = DeltaWeights*weights(genome1.genes, genome2.genes)
    return dd +dw < DeltaThreshold


def disjoint(genes1, genes2):
    i1 = {}
    for gene in genes1:
        i1[gene.innovation] = True

    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = True

    disjoint_genes = 0

    for gene in genes1:
        if gene.innovation not in i2.keys():
            disjoint_genes += 1
        elif not i2[gene.innovation]:
            disjoint_genes += 1

    for gene in genes2:
        if gene.innovation not in i1.keys():
            disjoint_genes += 1
        elif not i1[gene.innovation]:
            disjoint_genes += 1

    n = np.max([len(genes1), len(genes2)])

    return disjoint_genes / n


def weights(genes1, genes2):
    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = gene

    w_sum = 0
    coincident = 0
    # TODO for loop mutable check
    for gene in genes1:
        if gene.innovation in i2.keys():
            if i2[gene.innovation] != 0:
                gene2 = i2[gene.innovation]
                w_sum += np.abs(gene.weight - gene2.weight)
                coincident = coincident + 1

    if coincident == 0:
        return 0
    return w_sum / coincident


def start_pong_game():
    # We want a single frame update so there can be some input to the network
    pong = pong_pygame.Pong()
    pong.frame()
    pong.update_frame()
    return pong


def basic_genome():
    genome = Genome()
    innovation = 1
    genome.max_neuron = len(inputs)
    mutate(genome)
    # if len(genome.genes) == 0:
    #     print("Yep")
    return genome


def initialise_pool():
    global pool
    pool = Pool()
    for i in range(Population):
        basic = basic_genome()
        add_to_species(basic)

    initialise_run()


pong_game = start_pong_game()
pool = None
initialise_pool()

while True:
    """
    We'll need to change this species and genome selection.
    I'd like to randomly select two to compete and that we don't reselect those again.
    """
    for genome_index, genome in enumerate(pool.current_genomes_index):
        # Pygame image capture lags poorly if done every frame. Also don't want to be too erratic
        current_species = pool.species[pool.current_species_index[genome_index]]
        current_genome = current_species.genomes[genome]
        if pool.current_frame % 5 == 0:
            # TODO redesign this to work with multiple players
            buttons = evaluate_current_genome(current_genome)
            # Pressing buttons for next frame
            # print(f'Pressing {buttons}')
            pong_game.press_buttons(buttons, genome_index=genome_index, b_network=True)

        # Calculate fitness here
        # TODO check if this actually changes the fitness of genome in species or not
        current_genome.fitness = pool.current_frame

        if pong_game.is_completed:
            # player_index is this genome's player index. We need to do this for both players.
            if pong_game.completion_state[genome_index]:
                current_genome.fitness += 500

            if current_genome.fitness > pool.max_fitness:
                pool.max_fitness = current_genome.fitness
            current_species.genomes[genome].fitness = current_genome.fitness
    if pong_game.is_completed:
        print(f'Gen {pool.generation} | species seen {len(pool.seen_species)}/{len(pool.species)}, {pool.current_species_index} | genomes {pool.current_genomes_index}')
        # for i in range(2):
        #     current_species = pool.species[pool.current_species_index[i]]
        #     current_genome = current_species.genomes[pool.current_genomes_index[i]]
            # print(f'{current_species} {current_genome}')
        # print('\n')
        # Load the next genome for a new run
        # next_genomes()
        initialise_run()

    pool.current_frame = pool.current_frame + 1
    # print("A")
    pong_game.frame()
    pong_game.update_frame()
