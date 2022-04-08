import os
import time
import neat
import pygame
import random
import pickle
import visualize

pygame.font.init()


FLOOR = 730
WIN_WIDTH = 550
WIN_HEIGHT = 800

STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load("images/pipe.png").convert_alpha())
BG_IMAGE = pygame.transform.scale(pygame.image.load("images/bg.png").convert_alpha(), (600, 900))
BIRD_IMAGES = [pygame.transform.scale2x(pygame.image.load("images/bird" + str(x) + ".png")) for x in range(1, 4)]
BASE_IMAGE = pygame.transform.scale2x(pygame.image.load("images/base.png").convert_alpha())

gen = 0


class Bird(object):
    IMAGES = BIRD_IMAGES
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    tilt, tick_count, vel, image_count, img = \
        0, 0, 0, 0, IMAGES[0]

    def __init__(self, x: int, y):
        self.x = x
        self.y = y
        self.height = self.y

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        displacement = 16 if displacement >= 16 else displacement
        displacement -= 2 if displacement < 0 else 0

        self.y += displacement
        if displacement < 0 or self.y < self.height + 50:
            self.tilt = self.MAX_ROTATION if self.tilt < self.MAX_ROTATION else self.tilt
        else:
            self.tilt -= self.ROT_VEL if self.tilt > -90 else 0

    def draw(self, win):
        self.image_count += 1

        if self.image_count < self.ANIMATION_TIME:
            self.img = self.IMAGES[0]
        elif self.image_count < self.ANIMATION_TIME * 2:
            self.img = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME * 3:
            self.img = self.IMAGES[2]
        elif self.image_count < self.ANIMATION_TIME * 4:
            self.img = self.IMAGES[1]
        elif self.image_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMAGES[0]
            self.image_count = 0

        if self.tilt <= -80:
            self.img = self.IMAGES[1]
            self.image_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe(object):
    GAP = 200
    VEL = 5
    PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)
    PIPE_BOTTOM = PIPE_IMAGE

    height, top, bottom, passed =\
        0, 0, 0, False

    def __init__(self, x):
        self.x = x
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        mask = bird.get_mask()
        t = mask.overlap(pygame.mask.from_surface(self.PIPE_TOP), (self.x - bird.x, self.top - round(bird.y)))
        b = mask.overlap(pygame.mask.from_surface(self.PIPE_BOTTOM), (self.x - bird.x, self.bottom - round(bird.y)))
        return True if t or b else False


class Base(object):
    VEL = 5
    WIDTH = BASE_IMAGE.get_width()
    IMAGE = BASE_IMAGE

    x1, x2 = 0, WIDTH

    def __init__(self, y):
        self.y = y

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        self.x1 = self.x2 + self.WIDTH if self.x1 + self.WIDTH < 0 else self.x1
        self.x2 = self.x1 + self.WIDTH if self.x2 + self.WIDTH < 0 else self.x2

    def draw(self, win):
        win.blit(self.IMAGE, (self.x1, self.y))
        win.blit(self.IMAGE, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, generate, pipe_ind):
    generate = 1 if generate == 0 else generate

    win.blit(BG_IMAGE, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)

    for bird in birds:
        if DRAW_LINES:
            try:
                bird_pos = (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2)
                for pipe_pos in [
                    (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2, pipes[pipe_ind].height),
                    (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2, pipes[pipe_ind].bottom)
                ]:
                    pygame.draw.line(win, (255, 0, 0), bird_pos, pipe_pos, 5)
            except Exception as error:
                print('Error: ', error)
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), True, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    score_label = STAT_FONT.render("Gens: " + str(gen - 1), True, (255, 255, 255))
    win.blit(score_label, (10, 10))

    score_label = STAT_FONT.render("Alive: " + str(len(birds)), True, (255, 255, 255))
    win.blit(score_label, (10, 50))
    pygame.display.update()


def eval_genomes(genomes, config):
    global WIN, gen
    gen += 1

    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    still_run = True
    while still_run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                still_run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 1 if len(birds) > 0 and len(pipes) > 1 and \
                        birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width() else 0

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            bird.move()
            output = nets[birds.index(bird)].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()
        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)

    pygame.quit()
    quit()


def run():
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, os.path.join(os.path.dirname(__file__), 'config.txt'))
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    run()

