import pygame
import numpy as np

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
Lots of room for improvement to be a "better game" but will look into it after NN work.
"""


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.width = 25
        self.height = 150
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()

    # Move the sprite based on user keypresses
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -4)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 4)

        # Keep player on the screen
        if self.rect.top <= walls['top'].height:
            self.rect.top = walls['top'].height
        elif self.rect.bottom >= SCREEN_HEIGHT - walls['bottom'].height:
            self.rect.bottom = SCREEN_HEIGHT - walls['bottom'].height


class Wall(pygame.sprite.Sprite):
    def __init__(self, vertical=True):
        super(Wall, self).__init__()
        if vertical:
            self.height = SCREEN_HEIGHT
            self.width = 10
            self.surf = pygame.Surface((self.width, self.height))
            self.surf.fill((0, 0, 0))
            self.rect = self.surf.get_rect()
        else:
            self.height = 10
            self.width = SCREEN_WIDTH
            self.surf = pygame.Surface((self.width, self.height))
            self.surf.fill((255, 255, 255))
            self.rect = self.surf.get_rect()


class Projectile(pygame.sprite.Sprite):
    def __init__(self):
        super(Projectile, self).__init__()
        self.surf = pygame.Surface((25, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.velocity = [(-1)**(np.random.randint(0, 2))*4, (-1)**(np.random.randint(0, 2))*np.random.randint(1, 3)]

    def update(self):
        # self.rect.move_ip(0.5, 0.7)
        self.rect.move_ip(self.velocity)

        b_won = self.check_goal()
        if b_won:
            quit()
        self.check_rebound()

    def check_goal(self):
        if self.rect.left <= walls['left'].width:
            print("Player 2 wins")
            return True
        elif self.rect.right >= SCREEN_WIDTH - walls['right'].width:
            print("Player 1 wins")
            return True
        return False

    def check_rebound(self):
        if self.rect.top <= walls['top'].height:
            self.velocity = [self.velocity[0], -self.velocity[1]]
        elif self.rect.bottom >= SCREEN_HEIGHT - walls['bottom'].height:
            self.velocity = [self.velocity[0], -self.velocity[1]]

        if pygame.sprite.spritecollideany(self, players):
            self.velocity = [-self.velocity[0], self.velocity[1]]

def run_game():
    pygame.init()
    SCREEN_WIDTH = 750
    SCREEN_HEIGHT = 500
    CENTER = (int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2))
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    clock = pygame.time.Clock()

    player1 = Player()
    player2 = Player()
    players = pygame.sprite.Group()
    players.add(player1)
    players.add(player2)

    walls = {'left': Wall(vertical=True), 'right': Wall(vertical=True),
             'top': Wall(vertical=False), 'bottom': Wall(vertical=False)}
    walls['left'].rect.left = 0
    walls['right'].rect.right = SCREEN_WIDTH
    walls['top'].rect.top = 0
    walls['bottom'].rect.bottom = SCREEN_HEIGHT

    projectile = Projectile()
    projectile.rect.move_ip(CENTER)

    player1_loc = (SCREEN_WIDTH * 0.1, CENTER[1] - int(player1.height / 2))
    player2_loc = (SCREEN_WIDTH * 0.9 - player2.width, CENTER[1] - int(player2.height / 2))
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
        for wall in walls.values():
            screen.blit(wall.surf, wall.rect)

        # Draw projectile
        screen.blit(projectile.surf, projectile.rect)

        # Draw the player on the screen
        screen.blit(player1.surf, player1.rect)
        screen.blit(player2.surf, player2.rect)

        """
        Updating locations
        """
        pressed_keys = pygame.key.get_pressed()
        player1.update(pressed_keys)

        projectile.update()

        """
        ML Hook
        """
        pygame.image.save(screen, 'screen.png')

        """
        Updating frame
        """
        # Updates the display with a new frame
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
