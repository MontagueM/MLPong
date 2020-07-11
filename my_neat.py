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

while True:
