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

SCREEN_WIDTH = 750
SCREEN_HEIGHT = 500


class Player(pygame.sprite.Sprite):
    def __init__(self, pong_inst):
        super(Player, self).__init__()
        self.width = 25
        self.height = 150
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.pong_inst = pong_inst

    # Move the sprite based on user keypresses
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -4)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 4)

        # Keep player on the screen
        if self.rect.top <= self.pong_inst.walls['top'].height:
            self.rect.top = self.pong_inst.walls['top'].height
        elif self.rect.bottom >= SCREEN_HEIGHT - self.pong_inst.walls['bottom'].height:
            self.rect.bottom = SCREEN_HEIGHT - self.pong_inst.walls['bottom'].height


class Wall(pygame.sprite.Sprite):
    def __init__(self, pong_inst, vertical=True):
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
    def __init__(self, pong_inst):
        super(Projectile, self).__init__()
        self.surf = pygame.Surface((25, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.velocity = [(-1)**(np.random.randint(0, 2))*4, (-1)**(np.random.randint(0, 2))*np.random.randint(1, 3)]
        self.pong_inst = pong_inst

    def update(self):
        # self.rect.move_ip(0.5, 0.7)
        self.rect.move_ip(self.velocity)

        b_won = self.check_goal()
        if b_won:
            quit()
        self.check_rebound()

    def check_goal(self):
        if self.rect.left <= self.pong_inst.walls['left'].width:
            print("Player 2 wins")
            return True
        elif self.rect.right >= SCREEN_WIDTH - self.pong_inst.walls['right'].width:
            print("Player 1 wins")
            return True
        return False

    def check_rebound(self):
        if self.rect.top <= self.pong_inst.walls['top'].height:
            self.velocity = [self.velocity[0], -self.velocity[1]]
        elif self.rect.bottom >= SCREEN_HEIGHT - self.pong_inst.walls['bottom'].height:
            self.velocity = [self.velocity[0], -self.velocity[1]]

        if pygame.sprite.spritecollideany(self, self.pong_inst.players):
            self.velocity = [-self.velocity[0], self.velocity[1]]


class Pong:
    def __init__(self):
        pygame.init()
        CENTER = (int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2))
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.clock = pygame.time.Clock()

        self.player1 = Player(self)
        self.player2 = Player(self)
        self.players = pygame.sprite.Group()
        self.players.add(self.player1)
        self.players.add(self.player2)

        self.walls = {'left': Wall(self, vertical=True), 'right': Wall(self, vertical=True),
                 'top': Wall(self, vertical=False), 'bottom': Wall(self, vertical=False)}
        self.walls['left'].rect.left = 0
        self.walls['right'].rect.right = SCREEN_WIDTH
        self.walls['top'].rect.top = 0
        self.walls['bottom'].rect.bottom = SCREEN_HEIGHT

        self.projectile = Projectile(self)
        self.projectile.rect.move_ip(CENTER)

        player1_loc = (SCREEN_WIDTH * 0.1, CENTER[1] - int(self.player1.height / 2))
        player2_loc = (SCREEN_WIDTH * 0.9 - self.player2.width, CENTER[1] - int(self.player2.height / 2))
        self.player1.rect.move_ip(player1_loc)
        self.player2.rect.move_ip(player2_loc)

    def frame(self):
        """
        Drawing
        """
        self.screen.fill((0, 0, 0))

        # Draw borders
        for wall in self.walls.values():
            self.screen.blit(wall.surf, wall.rect)

        # Draw projectile
        self.screen.blit(self.projectile.surf, self.projectile.rect)

        # Draw the player on the screen
        self.screen.blit(self.player1.surf, self.player1.rect)
        self.screen.blit(self.player2.surf, self.player2.rect)

        """
        Updating locations
        """
        pressed_keys = pygame.key.get_pressed()
        self.player1.update(pressed_keys)

        self.projectile.update()

        """
        ML Hook
        """
        # pygame.image.save(self.screen, 'screen.png')

    def update_frame(self):
        # Updates the display with a new frame
        pygame.display.flip()
        self.clock.tick(60)

    def run_normal_pong(self):
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.frame()
            self.update_frame()

        pygame.quit()


if __name__ == "__main__":
    pong = Pong()
    pong.run_normal_pong()
