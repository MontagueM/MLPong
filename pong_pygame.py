import pygame

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

"""
pong plan:
- rectangular game space blocked by rect walls
- two controllable rect blockers
- circle for projectile
    - spawns with random direction const speed
    - deflects off walls with angle equal to incidence
    - if hit left or right walls despawns
"""


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.player_height = 150
        self.surf = pygame.Surface((25, self.player_height))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()


class Wall(pygame.sprite.Sprite):
    def __init__(self, vertical=True):
        super(Wall, self).__init__()
        if vertical:
            self.height = SCREEN_HEIGHT
            self.width = 20
            self.surf = pygame.Surface((self.width, self.height))
            self.surf.fill((0, 0, 0))
            self.wall = self.surf.get_rect()
        else:
            self.height = 20
            self.width = SCREEN_WIDTH
            self.surf = pygame.Surface((self.width, self.height))
            self.surf.fill((255, 255, 255))
            self.wall = self.surf.get_rect()


class Projectile(pygame.sprite.Sprite):
    def __init__(self):
        super(Projectile, self).__init__()
        self.surf = pygame.Surface((25, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()


pygame.init()
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 500
CENTER = (int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2))
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

player = Player()
walls = [Wall(vertical=True), Wall(vertical=True),
         Wall(vertical=False), Wall(vertical=False)]
projectile = Projectile()

running = True
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Draw borders
    screen.blit(walls[0].surf, (0, int((SCREEN_HEIGHT - walls[0].height)/2)))  # left
    screen.blit(walls[1].surf, (SCREEN_WIDTH-walls[0].width, int((SCREEN_HEIGHT - walls[1].height)/2)))  # right
    screen.blit(walls[2].surf, (int((SCREEN_WIDTH - walls[2].width) / 2), 0))  # top
    screen.blit(walls[3].surf, (int((SCREEN_WIDTH - walls[3].width) / 2), SCREEN_HEIGHT-walls[3].height))  # bottom
    # Draw projectile
    screen.blit(projectile.surf, CENTER)

    # Draw the player on the screen
    # screen.blit(player.surf, (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT - int(player.player_height/2)))


    # Updates the display with a new frame
    pygame.display.flip()


pygame.quit()
