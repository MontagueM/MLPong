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
        self.width = 25
        self.height = 150
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.pos = (0, 0)

    def set_pos(self, pos):
        self.pos = pos

    # Move the sprite based on user keypresses
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -1)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 1)

        # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT


class Wall(pygame.sprite.Sprite):
    def __init__(self, vertical=True):
        super(Wall, self).__init__()
        if vertical:
            self.height = SCREEN_HEIGHT
            self.width = 10
            self.surf = pygame.Surface((self.width, self.height))
            self.surf.fill((0, 0, 0))
            self.wall = self.surf.get_rect()
        else:
            self.height = 10
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
        self.pos = CENTER


pygame.init()
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 500
CENTER = (int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2))
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

player1 = Player()
player2 = Player()

walls = [Wall(vertical=True), Wall(vertical=True),
         Wall(vertical=False), Wall(vertical=False)]
walls[0].wall.left = 0
walls[1].wall.right = SCREEN_WIDTH
walls[2].wall.top = 0
walls[3].wall.bottom = SCREEN_HEIGHT

projectile = Projectile()

player1_loc = (SCREEN_WIDTH * 0.1, CENTER[1] - int(player1.height/2))
player2_loc = (SCREEN_WIDTH * 0.9 - player2.width, CENTER[1] - int(player2.height/2))
player1.rect.move_ip(player1_loc)
player2.rect.move_ip(player2_loc)

running = True
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    """
    Drawing
    """
    screen.fill((0, 0, 0))

    # Draw borders
    for wall in walls:
        screen.blit(wall.surf, wall.wall)

    # Draw projectile
    screen.blit(projectile.surf, CENTER)

    # Draw the player on the screen
    screen.blit(player1.surf, player1.rect)
    screen.blit(player2.surf, player2.rect)

    """
    Key presses
    """
    pressed_keys = pygame.key.get_pressed()
    player1.update(pressed_keys)

    """
    Other stuff
    """
    # Updates the display with a new frame
    pygame.display.flip()


pygame.quit()
