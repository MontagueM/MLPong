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
        - randomly set weights as usual
        - run for like 1000 games
        - pick the players who won the game and train using their weights and stuff
        - repeat again and again for different evolutions?
"""