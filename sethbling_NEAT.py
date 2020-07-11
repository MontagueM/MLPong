"""
Pool: class that stores all the overall information (one pool is like one run of a simulation)
Generation: a single static group of species and genomes. Each generation, its species are bred and culled to further evolution.
Species: a group of genomes that
Genome: a specific ANN that is run to find its fitness
Gene: a unit within a genome (named as such for the evolution link)
Fitness: a measurement statistic that rates how good a specific genome (and so a species too) is.
Stale: an unimportant unit or species that is not making progress
"""


#####
import numpy as np

"""
Using NEAT (NeuroEvolution of Augmenting Topologies)
Paper:
http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
"""

# InputSize = (BoxRadius * 2 + 1) * (BoxRadius * 2 + 1)
# Inputs = InputSize + 1
inputs = ?  # The input data SIZE
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
    pool['innovation'] = pool['innovation'] + 1
    return pool['innovation']

class Pool:
    def __init__(self):
        self.species = []
        self.generation = 0
        self.innovation = outputs
        self.current_species = 1
        self.current_genome = 1
        self.current_frame = 0
        self.max_fitness = 0


class Species:
    def __init__(self):
        self.top_fitness = 0
        self.staleness = 0
        self.genomes = []
        self.average_fitness = 0


class Genome:
    def __init__(self):
        self.genes = []
        self.adjusted_fitness = 0
        self.network = []
        self.max_neuron = 0
        self.global_rank = 0
        self.mutation_rates = MutationRates()


class MutationRates:
    def __init__(self):
        self.connections = MutateConnectionsChance
        self.link = LinkMutationChance
        self.bias = BiasMutationChance
        self.node = NodeMutationChance
        self.enable = EnableMutationChance
        self.disable = DisableMutationChance
        self.step = StepSize



def new_pool():
    #TODO Refactor
    pool = Pool()
    return pool


def new_species():
    #TODO Refactor
    species = Species()

    return species

def copy_genome(genome):
    #TODO Refactor
    return dict(genome)


def basic_genome():
    #TODO Refactor
    genome = Genome()
    innovation = 1
    genome['max_neuron'] = inputs
    mutate(genome)

    return genome

class Gene:
    def __init__(self):
        self.into = 0
        self.out = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0

def new_gene():
    gene = Gene()

    return gene

def copy_gene(gene):
    return dict(gene)

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

    for inp in range(inputs):
        network.neurons[inp] = Neuron()

    for out in range(len(outputs)):
        network.neurons[MaxNodes+out] = Neuron()

    # table.sort(genome.genes, function(a, b)
    # return (a.out < b.out)
    # end)
    _, genome.genes = zip(*sorted(zip([x.out for x in genome.genes], genome.genes)))

    for gene in genome.genes:
        if gene.enabled:
            if network.neurons[gene.out] == 0:
                network.neurons[gene.out] = Neuron()
            neuron = network.neurons[gene.out]
            neuron.incoming.append(gene)
            if network.neurons[gene.into] == 0:
                network.neurons[gene.into] = Neuron()

    genome.network = network

def evaluate_network(network, inputs):
    inputs += 1
    # TODO check this line
    if inputs != len(inputs):
        print("Incorrect number of neural network inputs.")
        return

    for i in inputs:
     network.neurons[i].value = inputs[i]

    for _, neuron in pairs(network.neurons):
        sum_ = 0
        for incoming in neuron.incoming:
            other = network.neurons[incoming.into]
            sum_ += incoming.weight * other.value

        if len(neuron.incoming) > 0:
            neuron.value = sigmoid(sum_)

    button_outputs = {}
    for o in range(len(outputs)):
        if network.neurons[MaxNodes+o].value > 0:
            button_outputs[outputs[o]] = True
        else:
            button_outputs[outputs[o]] = False

    return button_outputs


def crossover(g1, g2):
    # Make sure g1 is the higher fitness genome
    if g2.fitness > g1.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg

    child = new_genome()

    innovations2 = {}
    for gene in g2.genes:
        innovations2[gene.innovation] = gene

    for gene1 in g1.genes:
        gene2 = innovations2[gene1.gene.innovation]
        if gene2 != 0 and np.random.randint(2) == 1 and gene2.enabled:
            child.genes.append(copy_gene(gene2))
        else:
            child.genes.append(copy_gene(gene1))

    child.maxneuron = np.max(g1.maxneuron, g2.maxneuron)

    for mutation, rate in pairs(g1.mutationRates):
        child.mutationRates.mutation = rate

    return child


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
    for _, _ in pairs(neurons):
        count += 1

    # TODO check this outcome is same as lua's math.random()
    n = np.random.randint(1, count)

    for k,v in pairs(neurons):
        n -= 1
        if n == 0:
            return k

    return 0


def contains_link(genes, link):
    for gene in genes:
        if gene.into == link.into and gene.out == link.out:
            return True


def point_mutate(genome):
    step = genome.mutation_rates.step

    for gene in genes:
        if np.random.random() < PerturbChance:
            gene.weight += np.random.random() * step * 2 - step
        else:
            gene.weight = np.random.random() * 4 - 2


def link_mutate(genome, force_bias):
    neuron1 = random_neuron(genome.genes, False)
    neuron2 = random_neuron(genome.genes, True)

    new_link = new_gene()
    if neuron1 <= len(inputs) and neuron2 <= len(inputs):
        # Both input nodes
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
        return

    new_link.innovation = new_innovation()
    new_link.weight = np.random.random() * 4 - 2
    genome.genes.append(new_link)


def node_mutate(genome):
    if len(genome.genes) == 0:
        return

    genome.max_neuron += 1

    gene = genome.genes[np.random.randint(1, len(genome.genes))]

    if not gene.enabled:
        return

    gene.enabled = False

    gene1 = copy_gene(gene)
    gene1.out = genome.max_neuron
    gene1.weight = 1
    gene1.innovation = new_innovation()
    gene1.enabled = True
    genome.genes.append(gene1)

    gene2 = copy_gene(gene)
    gene2.out = genome.max_neuron
    gene2.innovation = new_innovation()
    gene2.enabled = True
    genome.genes.append(gene2)


def enable_disable_mutate(genome, enable):
    candidates = []
    for _, gene in pairs(genome.genes):
        if gene.enabled != enable:
            candidates.append(candidates, gene)

    if len(candidates) == 0:
        return

    gene = candidates[np.random.random(1, len(candidates))]
    gene.enabled = not gene.enabled


def mutate(genome):
    for mutation, rate in pairs(genome.mutation_rates):
        if np.random.randint(2) == 1:
            genome.mutation_rates.mutation = 0.95*rate
        else:
            genome.mutation_rates.mutation = 1.05263*rate

    if np.random.random() < genome.mutation_rates.connections:
        point_mutate(genome)


    while genome.mutation_rates.link > 0:
        if np.random.random() > genome.mutation_rates.link:
            link_mutate(genome, False)

        genome.mutation_rates.link -= 1

    while genome.mutation_rates.bias > 0:
        if np.random.random() > genome.mutation_rates.bias:
            link_mutate(genome, true)

        genome.mutation_rates.bias -= 1

    while genome.mutation_rates.node > 0:
        if np.random.random() > genome.mutation_rates.node:
            node_mutate(genome)

        genome.mutation_rates.node -= 1

    while genome.mutation_rates.enable > 0:
        if np.random.random() > genome.mutation_rates.enable:
            enable_disable_mutate(genome, True)

        genome.mutation_rates.enable -= 1

    while genome.mutation_rates.disable > 0:
        if np.random.random() > genome.mutation_rates.disable:
            enable_disable_mutate(genome, False)

        genome.mutation_rates.disable -= 1

def disjoint(genes1, genes2):
    i1 = {}
    for gene in genes1:
        i1[gene.innovation] = True

    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = True

    disjoint_genes = 0

    for gene in genes1:
        if not i2[gene.innovation]:
            disjoint_genes += 1

    for gene in genes1:
        if not i1[gene.innovation]:
            disjoint_genes += 1

    n = np.max(len(genes1), len(genes2))

    return disjoint_genes / n


def weights(genes1, genes2):
    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = gene

    sum_ = 0
    coincident = 0
    for gene in genes1:
        if i2[gene.innovation] != 0:
            gene2 = i2[gene.innovation]
            sum_ += np.abs(gene.weight - gene2.weight)
            coincident = coincident + 1

    return sum_ / coincident


def same_species(genome1, genome2):
    dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
    dw = DeltaWeights*weights(genome1.genes, genome2.genes)
    return dd +dw < DeltaThreshold


def rank_globally():
    glob = []
    for species in pool.species:
        for g in species.genomes:
            glob.append(g)

    #         table.sort(global, function (a,b)
    #                 return (a.fitness < b.fitness)
    #         end)
    _, glob = zip(*sorted(zip([x.fitness for x in glob], glob)))

    for g in glob:
        g.global_rank = g


def calculate_average_fitness(species):
    total = 0
    for genome in species.genomes:
        total += genome.global_rank

    species.average_fitness = total / len(species.genomes)


def total_average_fitness():
    total = 0
    for species in pool.species:
        total += species.average_fitness

    return total


def cull_species(cut_to_one):
    for species in pool.species:
        #                 table.sort(species.genomes, function (a,b)
        #                         return (a.fitness > b.fitness)
        #                 end)
        _, species.genomes = zip(*sorted(zip([x.fitness for x in species.genomes], species.genomes)))

    remaining = np.ceil(len(species.genomes)/2)

    if cut_to_one:
        remaining = 1

    while len(species.genomes) > remaining:
        # TODO check this line works as intended
        species.genomes.pop()


def breed_child(species):
    if np.random.random() < CrossoverChance:
        g1 = species.genomes[np.random.random(len(species.genomes))]
        g2 = species.genomes[np.random.random(len(species.genomes))]
        child = crossover(g1, g2)
    else:
        g = species.genomes[np.random.random(len(species.genomes))]
        child = copy_gene(g)

    mutate(child)

    return child


def remove_stale_species():
    survived = []
    for species in pool.species:
        _, species.genomes = zip(*sorted(zip([x.fitness for x in species.genomes], species.genomes)))

        if species.genomes[1].fitness > species.top_fitness:
            species.top_fitness = species.genomes[1].fitness
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


def add_to_species(child):
    found_species = False
    for species in pool.species:
        if not found_species and same_species(child, species.genomes[1]):
            species.genomes.append(child)
            found_species = True

    if not found_species:
        child_species = new_species()
        child_species.genomes.append(child)
        pool.species.append(child_species)


def new_generation():
    cull_species(False)  # Cull the bottom half of each species
    rank_globally()
    remove_stale_species()
    rank_globally()
    for species in pool.species:
        calculate_average_fitness(species)
    remove_weak_species()
    sum_ = total_average_fitness()
    children = []
    for species in pool.species:
        breed = np.floor(species.average_fitness / sum_ * Population) - 1
        for i in range(breed):
            children.append(breed_child(species))

    cull_species(True)  # Cull all but top member of each species
    while len(children) + len(pool.species) < Population:
        species = pool.species[np.random.random(len(pool.species))]
        children.append(breed_child(species))

    for child in children:
        add_to_species(child)


def initialise_pool():
    pool = Pool()
    for i in range(Population):
        basic = basic_genome()
        add_to_species(basic)

    initialise_run()


def initialise_run():
    rightmost = 0
    pool.current_frame = 0
    timeout = TimeoutConstant
    clear_joypad()
    species = pool.species[pool.current_species]
    genome = species.genome[pool.current_genome]
    generate_network(genome)
    evaluate_current()


def evaluate_current():
    species = pool.species[pool.current_species]
    genome = species.genomes[pool.current_genome]

    inputs = get_inputs()
    controller = evaluate_network(genome.network, inputs)

    if controller['Up'] and controller['Down']:
        controller['Up'] = False
        controller['Down'] = False

if pool == 0:
    initialise_pool()


def next_genome():
    pool.current_genome += 1
    if pool.current_genome > len(pool.species[pool.current_species].genomes):
        pool.current_genome = 1
        pool.current_species += 1
        if pool.current_species > len(pool.species):
            new_generation()
            pool.current_species = 1


def fitness_already_measured():
    species = pool.species[pool.current_species]
    genome = species.genomes[pool.current_genome]

    return genome.fitness != 0

while True:
    species = pool.species[pool.current_species]
    genome = species.genomes[pool.current_genome]

    if pool.current_frame % 5 == 0:
        evaluate_current()

    # Calculate fitness here
    if completed_run:
        genome.fitness = fitness

        if fitness > pool.max_fitness:
            pool.max_fitness = fitness

        print(f'Gen {pool.generation} species {pool.current_species} genome {pool.current_genome}')

        pool.current_species = 1
        pool.current_genome = 1
        while fitness_already_measured():
            next_genome()
        initialise_run()

    pool.currentFrame = pool.currentFrame + 1

    # Frame advance the game