import numpy as np
import pong_pygame

"""
We need to implement:
- computer vision?
    - could instead just use the numbers of projectile location, velocity, etc.
    - should try first with computer vision
- a way to play the pong game very quickly (velocity multiplier)
- a way to randomise inputs
- an overall methodology of driving the pong game at the same time as using neural networking inputs
    - this methodology likely is:
        - run the game with player1, player2
        - feed in as inputs the computer vision stuff
        - randomly set weights as usual (need to pick a classifier thing, I'm prob not using my ANN)
        - run for like 1000 games
        - pick the players with greater fitness score and train using their weights and stuff (or breeding?)
        - repeat again and again for different evolutions?

Try figure it out myself and then look at something after.

- make an ANN for N genomes, all randomised
- save an image of the current game state
- use computer vision to convert into inputs
- send inputs through the corresponding genome
- take outputs and place back into pygame
- repeat until game ends
- cull the weakest genomes in terms of fitness
- breed the ones left over to produce another evolution with same number of genomes as prev evolution
"""

"""
First try without any sprite recognition.
If it doesn't work well could try manual sprite recognition.
"""

####
# Actual procedure
"""
Procedure:
--------------
Make a new pool
    Populate each species with a single basic genome
Initialise the run for a specific genome
    Clear the buttons for new inputs
    Generate the network for this genome
        Populate the neuron array with inputs and output units
        Sort genome genes (neuron) by their connection to outputs?
        If gene is enabled
            // Just to check that we have no unconnected neurons (genes)
            If gene has no output
                Add a new neuron to array
            If it has no input
                Add a new neuron to array 
If run has not been completed:
    Evaluate the current network
        For every input
            Set the input neuron values to inputs
        For every neuron in network
            For all incoming neurons to this neuron
                Sum all incoming connection weight * value of that incoming neuron
            Neuron value = sigmoid(sum)
        If the output value for each button is > 0
            Activate that button as a press next time
    Set buttons to be pressed for next frame
    Progress the game by a single frame
If run has been completed:
    Calculate fitness
        // TEMP
        Reward how long the game goes on for eg 1 point per ms (can be reached via current_frame)
        Reward winning eg 100 for a win (reached via game return)
    If this genome's fitness is the highest in the pool
        Set the pool's max fitness to this new fitness (this is just for checking its working)
    If current genome fitness != 0
        Go to next genome
            Add one to current genome counter
            If current genome counter > num current genomes in species
                Current genome counter = 1
                Add one to current species counter
                If current species counter > num current species in generation
                    Current species counter = 1
                    Create new generation
                    // The logic for this is below as its quite a lot
                    
    Initialise a new run with the new genome/species/generation
// This script will run forever until you stop it (can add a max generation)
// Should add a write ability for each genome
"""

"""
Create new generation
    Cull the bottom half of each species in this generation
    Remove stale species
    Rank each species globally
    For each species in pool
        Calculate the average fitness for that species
    Remove weak species based on fitness average
    Calculate total average fitness as a variable
    Children init var []
    For each species in pool
        // This calc means that if this species fitness is higher we get more breeding from it
        Number of neurons to breed from this species = floor(species average fitness / total average fitness * Population) - 1
        For number of neurons to breed
            Breed a child from this species
            Add this bred child to children array
    Cull all but top genome of each species
    While number of children < Population
        Breed a child from this species (which is now only the best performer)
        Add this bred child to children array
    For every child in children
        Add child to species
            ...
    
    ...
"""

"""
Breeding
    If prob within crossover chance (0.75)
        Select two random genomes from species to breed between
        Create a child from the crossover of these two
            Decide 
    Else
        Copy a random gene and make a child
    
    Mutate the child
        Randomly slightly change each attr for this genomes mutation rates
        If prob within connections mutation rates
            Mutate weights
                ...
        .
        .
        .
"""

#######

"""
I need to write a system that works with two genomes being trained at the same time.
These two should probably be randomly selected and not specified as related in any way.
"""

#######

inputs = ?  # The input data SIZE
outputs = ['K_UP', 'K_DOWN']

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
    pool.innovation = pool.innovation + 1
    return pool.innovation

class Pool:
    def __init__(self):
        self.species = []
        self.seen_species = []
        self.generation = 0
        # Find out if innovations are required
        self.innovation = outputs
        self.current_species = [-1, -1]
        self.current_genomes = [-1, -1]
        self.current_frame = 0
        self.max_fitness = 0


class Species:
    def __init__(self):
        self.top_fitness = 0
        self.staleness = 0
        self.genomes = []
        self.seen_genomes = []
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

    _, genome.genes = zip(*sorted(zip([x.out for x in genome.genes], genome.genes)))

    for gene in genome.genes:
        if gene.enabled:
            if network.neurons[gene.out] == 0:
                network.neurons[gene.out] = Neuron()
            neuron = network.neurons[gene.out]
            neuron.incoming.append(gene)
            if network.neurons[gene.into] == 0:
                network.neurons[gene.into] = Neuron()

    # TODO check that this works in setting the network
    genome.network = network


def rank_globally():
    glob = []
    for species in pool.species:
        for g in species.genomes:
            glob.append(g)

    _, glob = zip(*sorted(zip([x.fitness for x in glob], glob)))

    for g in glob:
        # TODO make sure this actually sets it (pointer vs reference)
        g.global_rank = g


def calculate_average_fitness(species):
    total = 0
    # TODO check this total works
    for gen in species.genomes:
        total += gen.global_rank

    species.average_fitness = total / len(species.genomes)


def total_average_fitness():
    total = 0
    # TODO check this total works
    for species in pool.species:
        total += species.average_fitness

    return total


def remove_stale_species():
    survived = []
    for species in pool.species:
        _, species.genomes = zip(*sorted(zip([x.fitness for x in species.genomes], species.genomes)))

        if species.genomes[1].fitness > species.top_fitness:
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
    # TODO check setting works
    pool.species = survived


def cull_species(cull_to_one):
    for i, sp in enumerate(pool.species):
        _, sp.genomes = zip(*sorted(zip([x.fitness for x in sp.genomes], sp.genomes)))

        remaining = np.ceil(len(sp.genomes)/2)

        if cull_to_one:
            remaining = 1

        pool.species[i].genomes = sp.genomes[-remaining:]


def evaluate_network(network, inputs):
    inputs += 1
    # TODO check this line
    if inputs != len(inputs):
        print("Incorrect number of neural network inputs.")
        return

    for i in inputs:
     network.neurons[i].value = inputs[i]

    for _, neuron in network.neurons.__dict__.items():
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


def breed_child(species):
    if np.random.random() < CrossoverChance:
        g1 = species.genomes[np.random.random(len(species.genomes))]
        g2 = species.genomes[np.random.random(len(species.genomes))]
        child = crossover(g1, g2)
    else:
        g = species.genomes[np.random.random(len(species.genomes))]
        child = g

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
        gene2 = innovations2[gene1.gene.innovation]
        if gene2 != 0 and np.random.randint(2) == 1 and gene2.enabled:
            child.genes.append(gene2)
        else:
            child.genes.append(gene1)

    child.maxneuron = np.max(g1.maxneuron, g2.maxneuron)

    for mutation, rate in pairs(g1.mutationRates):
        child.mutationRates.mutation = rate

    return child


def mutate(genome):
    """
    pairs(table) in lua produces an iterator of table key:value
    so the below of pairs(genome.mutation_rates) will produce an iterator of
    connections, MutateConnectionsChance
    link, LinkMutationChance
    bias, BiasMutationChance
    .
    .
    .

    The equivalent of this in Python for a class:
    """
    for mutation, rate in genome.mutation_rates.__dict__.items():
        if np.random.randint(2) == 1:
            setattr(genome.mutation_rates, mutation, 0.95*rate)
        else:
            setattr(genome.mutation_rates, mutation, 1.05263*rate)

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

    gene1 = gene
    gene1.out = genome.max_neuron
    gene1.weight = 1
    gene1.innovation = new_innovation()
    gene1.enabled = True
    genome.genes.append(gene1)

    gene2 = gene
    gene2.out = genome.max_neuron
    gene2.innovation = new_innovation()
    gene2.enabled = True
    genome.genes.append(gene2)


def enable_disable_mutate(genome, enable):
    candidates = []
    for _, gene in genome.genes.__dict__.items():
        if gene.enabled != enable:
            candidates.append(gene)

    if len(candidates) == 0:
        return

    gene = candidates[np.random.randint(len(candidates))]
    gene.enabled = not gene.enabled


def evaluate_current():
    # TODO Fix
    species = pool.species[pool.current_species]
    genome = species.genomes[pool.current_genome]

    # TODO add inputs from computer vision
    inputs = get_inputs()
    controller = evaluate_network(genome.network, inputs)

    if controller['K_UP'] and controller['K_DOWN']:
        controller['K_UP'] = False
        controller['K_DOWN'] = False

    return controller


def initialise_run():
    pool.current_frame = 0
    # Resetting buttons
    pong_game.press_buttons([{'K_UP': False, 'K_DOWN': False},
                            {'K_LEFT': False, 'K_RIGHT': False}], b_network=True)
    # Load the next genome for a new run
    next_genomes()

    for i in range(2):
        species = pool.species[pool.current_species[i]]
        genome = species.genomes[pool.current_genomes[i]]
        generate_network(genome)
    evaluate_current()


def next_genomes():
    for i in range(2):
        # Selecting two random genomes from random species, making sure that we don't
        # select the same one again (trying to be efficient about it)
        reduced_species = list(set(pool.species) - set(pool.seen_species))
        red_species = reduced_species[np.random.randint(len(reduced_species))]
        reduced_genomes = list(set(red_species.genomes) - set(red_species.genomes))

        pool.current_genomes[i] = reduced_genomes[np.random.randint(len(reduced_genomes))]

        # Making sure that we mark the species as seen if all genomes in it are seen
        pool.current_species[i] = pool.species.index(red_species)
        pool_species = pool.species[pool.current_species[i]]
        pool_species.seen_genomes.append(pool.current_genomes[i])

        if set(pool_species.seen_genomes) == set(pool_species.genomes):
            pool.seen_species.append(red_species)
        if set(pool.species) == set(pool.seen_species):
            new_generation()
            next_genomes()
    print(pool.current_genomes, pool.current_species)


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
        breed = np.floor(species.average_fitness / sum_ * Population) - 1
        for i in range(breed):
            children.append(breed_child(species))

    cull_species(cull_to_one=True)  # Cull all but top member of each species
    while len(children) + len(pool.species) < Population:
        species = pool.species[np.random.random(len(pool.species))]
        children.append(breed_child(species))

    for child in children:
        add_to_species(child)


def add_to_species(child):
    found_species = False
    for species in pool.species:
        if not found_species and same_species(child, species.genomes[0]):
            species.genomes.append(child)
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


def start_pong_game():
    # We want a single frame update so there can be some input to the network
    pong = pong_pygame.Pong()
    pong.frame()
    pong.update_frame()
    return pong

pool = Pool()
pong_game = start_pong_game()

while True:
    """
    We'll need to change this species and genome selection.
    I'd like to randomly select two to compete and that we don't reselect those again.
    """

    for genome_index, genome in enumerate(pool.current_genomes):
        # Pygame image capture lags poorly if done every frame. Also don't want to be too erratic
        if pool.current_frame % 5 == 0:
            # TODO redesign this to work with multiple players
            buttons = evaluate_current()
            # Pressing buttons for next frame
            pong_game.press_buttons(buttons, b_network=True)

        # Calculate fitness here
        # TODO check if this actually changes the fitness of genome in species or not
        genome.fitness = pool.current_frame

        if pong_game.is_completed:
            # player_index is this genome's player index. We need to do this for both players.
            if pong_game.completion_state[genome_index]:
                genome.fitness += 500

            if genome.fitness > pool.max_fitness:
                pool.max_fitness = genome.fitness

    if pong_game.is_completed:
        print(f'Gen {pool.generation} species {pool.current_species} genome {pool.current_genomes}')

        initialise_run()

    pool.currentFrame = pool.currentFrame + 1