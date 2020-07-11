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

"""