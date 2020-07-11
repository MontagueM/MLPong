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
    def __init__(self, pong_inst, player_index):
        super(Player, self).__init__()
        self.width = 25
        self.height = 150
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
        self.pong_inst = pong_inst
        self.player_index = player_index

    # Move the sprite based on user keypresses
    def update(self, pressed_keys, b_network=False):
        if b_network:
            if pressed_keys['Up']:
                self.rect.move_ip(0, -4)
            if pressed_keys['Down']:
                self.rect.move_ip(0, 4)
        else:
            if self.player_index:
                buttons = [K_LEFT, K_RIGHT]
            else:
                buttons = [K_UP, K_DOWN]
            if pressed_keys[buttons[0]]:
                self.rect.move_ip(0, -4)
            if pressed_keys[buttons[1]]:
                self.rect.move_ip(0, 4)

        # Keep player on the screen
        if self.rect.top <= self.pong_inst.walls['top'].height:
            self.rect.top = self.pong_inst.walls['top'].height
        elif self.rect.bottom >= SCREEN_HEIGHT - self.pong_inst.walls['bottom'].height:
            self.rect.bottom = SCREEN_HEIGHT - self.pong_inst.walls['bottom'].height


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

        self.pong_inst.is_completed = self.check_goal()
        if self.pong_inst.is_completed:
            return
        self.check_rebound()

    def check_goal(self):
        if self.rect.left <= self.pong_inst.walls['left'].width:
            # print("Player 2 wins")
            self.pong_inst.completion_state = [False, True]
            return True
        elif self.rect.right >= SCREEN_WIDTH - self.pong_inst.walls['right'].width:
            # print("Player 1 wins")
            self.pong_inst.completion_state = [True, False]
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

        self.player0 = Player(self, 0)
        self.player1 = Player(self, 1)
        self.players = pygame.sprite.Group()
        self.players.add(self.player0)
        self.players.add(self.player1)

        self.walls = {'left': Wall(vertical=True), 'right': Wall(vertical=True),
                 'top': Wall(vertical=False), 'bottom': Wall(vertical=False)}
        self.walls['left'].rect.left = 0
        self.walls['right'].rect.right = SCREEN_WIDTH
        self.walls['top'].rect.top = 0
        self.walls['bottom'].rect.bottom = SCREEN_HEIGHT

        self.projectile = Projectile(self)
        self.projectile.rect.move_ip(CENTER)

        player0_loc = (SCREEN_WIDTH * 0.1, CENTER[1] - int(self.player0.height / 2))
        player2_loc = (SCREEN_WIDTH * 0.9 - self.player1.width, CENTER[1] - int(self.player1.height / 2))
        self.player0.rect.move_ip(player0_loc)
        self.player1.rect.move_ip(player2_loc)

        self.is_completed = False
        self.completion_state = []

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
        self.screen.blit(self.player0.surf, self.player0.rect)
        self.screen.blit(self.player1.surf, self.player1.rect)

        """
        Updating locations
        """
        self.projectile.update()

    def press_buttons(self, buttons, genome_index=-1, b_network=False):
        if b_network:
            if genome_index == 0:
                self.player0.update(buttons[0], b_network=b_network)
            else:
                self.player1.update(buttons[1], b_network=b_network)
        else:
            self.player0.update(buttons, b_network=b_network)
            self.player1.update(buttons, b_network=b_network)

    def capture_screen(self):
        # Only capture every 5 frames or more as slow method
        pygame.image.save(self.screen, 'screen.png')

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
            if self.is_completed:
                running = False
            self.frame()
            self.update_frame()
            pressed_keys = pygame.key.get_pressed()
            self.press_buttons(pressed_keys)
        pygame.quit()


if __name__ == "__main__":
    pong = Pong()
    pong.run_normal_pong()
