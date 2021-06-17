# -------------------------------------------------------------------------------------------------
# MIT License
# If you use this in something awesome, credit would be nice but isn't required
# Written by Keilan
# Website: www.scholtek.com
# Github: https://github.com/Keilan
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------
import os
import sys
import random
from copy import deepcopy
import multiprocessing
import jsonpickle

import numpy
from PIL import Image, ImageDraw

# -------------------------------------------------------------------------------------------------
# Knobs and Dials
# -------------------------------------------------------------------------------------------------
# from docutils.utils.math.math2html import file

POP_SIZE = 50
POP_PER_GENERATION = 50
MUTATION_CHANCE = 0.1
CROSSOVER_CHANCE = 0.6
ADD_GENE_CHANCE = 0.3
REM_GENE_CHANCE = 0.2
INITIAL_GENES = 50
TOURNAMENT_SIZE = 5
# todo - 200

# How often to output images and save files
GENERATIONS_PER_IMAGE = 1000
GENERATIONS_PER_SAVE = 1000

print("XOVER_MUTATE_SELECT_RANDOM")
try:
    globalTarget = Image.open("reference.png")
except IOError:
    print("File reference.png must be located in the same directory as poly_artist.py.")
    exit()


# -------------------------------------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------------------------------------
class Point:
    """
    A 2D point. You can add them together if you want.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, o):
        return Point(self.x + o.x, self.y + o.y)


class Color:
    """
    A color. You can shift it by a given value.
    """

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def shift(self, r, g, b):
        self.r = max(0, min(255, self.r + r))
        self.g = max(0, min(255, self.g + g))
        self.b = max(0, min(255, self.b + b))

    def __str__(self):
        return "({},{},{})".format(self.r, self.g, self.b)


# -------------------------------------------------------------------------------------------------
# Genetic Classes
# -------------------------------------------------------------------------------------------------
class Gene:
    """
    A gene is the object that can be mutated. Genetic algorithms work by randomly mutating genes
    and then using some function to determine how "ideal" the resulting organism is.

    This one is basically a circle, with a size, color, and position on the canvas.
    """

    def __init__(self, size):
        self.size = size  # The canvas size so we know the maximum position value

        self.diameter = random.randint(1,10)
        self.pos = Point(random.randint(0, size[0]), random.randint(0, size[1]))
        self.color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.params = ["diameter", "pos", "color"]

    def mutate(self):
        # Decide how big the mutation will be
        mutation_size = max(1, int(round(random.gauss(10, 4)))) / 100

        # Decide what will be mutated
        mutation_type = random.choice(self.params)

        # Mutate the thing
        if mutation_type == "diameter":
            self.diameter = max(1, random.randint(int(self.diameter * (1 - mutation_size)),
                                                  int(self.diameter * (1 + mutation_size))))

        elif mutation_type == "pos":
            x = max(0, random.randint(int(self.pos.x * (1 - mutation_size)), int(self.pos.x * (1 + mutation_size))))
            y = max(0, random.randint(int(self.pos.y * (1 - mutation_size)), int(self.pos.y * (1 + mutation_size))))
            self.pos = Point(min(x, self.size[0]), min(y, self.size[1]))

        elif mutation_type == "color":
            r = min(max(0, random.randint(int(self.color.r * (1 - mutation_size)),
                                          int(self.color.r * (1 + mutation_size)))), 255)
            g = min(max(0, random.randint(int(self.color.g * (1 - mutation_size)),
                                          int(self.color.g * (1 + mutation_size)))), 255)
            b = min(max(0, random.randint(int(self.color.b * (1 - mutation_size)),
                                          int(self.color.b * (1 + mutation_size)))), 255)
            self.color = Color(r, g, b)

    def getSave(self):
        """
        Allows us to save an individual gene in case the program is stopped.
        """
        so = {}
        so["size"] = self.size
        so["diameter"] = self.diameter
        so["pos"] = (self.pos.x, self.pos.y)
        so["color"] = (self.color.r, self.color.g, self.color.b)
        return so

    def loadSave(self, so):
        """
        Allows us to load an individual gene in case the program is stopped.
        """
        self.size = so["size"]
        self.diameter = so["diameter"]
        self.pos = Point(so["pos"][0], so["pos"][1])
        self.color = Color(so["color"][0], so["color"][1], so["color"][2])


class Organism:
    """
    The organism consists of a variety of genes that work together for some sort of effect. The main
    effect of the genetic algorithm takes place here, as each step involves mutating some of the
    organisms genes to produce offspring, and the best performing of those offspring carries on.

    This organism contains a bunch of genes that draw circles, working together to draw a picture.
    """

    def __init__(self, size, num):
        self.size = size

        # Create random genes up to the number given
        self.genes = [Gene(size) for _ in range(num)]

    def mutate(self):
        # For small numbers of genes, each one has a random chance of mutating
        if len(self.genes) < 200:
            for g in self.genes:
                if random.random() < MUTATION_CHANCE:
                    g.mutate()

        # For large numbers of genes, pick a random sample, this is statistically equivalent and faster
        else:
            for g in random.sample(self.genes, int(len(self.genes) * MUTATION_CHANCE)):
                g.mutate()

        # We also have a chance to add or remove a gene
        if ADD_GENE_CHANCE < random.random():
            self.genes.append(Gene(self.size))
        if len(self.genes) > 0 and REM_GENE_CHANCE < random.random():
            self.genes.remove(random.choice(self.genes))

    def crossover(self, other):
        if CROSSOVER_CHANCE > random.random():
            index = random.randint(1, len(self.genes) - 1)
            o1 = deepcopy(self)
            o1.genes = self.genes[0:index] + other.genes[index:]
            o2 = deepcopy(self)
            o2.genes = other.genes[0:index] + self.genes[index:]
            return o1, o2

    def drawImage(self):
        """
        Using the Image module, use the genes to draw the image.
        """
        image = Image.new("RGB", self.size, (255, 255, 255))
        canvas = ImageDraw.Draw(image)

        for g in self.genes:
            color = (g.color.r, g.color.g, g.color.b)
            canvas.ellipse([g.pos.x - g.diameter, g.pos.y - g.diameter, g.pos.x + g.diameter, g.pos.y + g.diameter],
                           outline=color, fill=color)

        return image

    def getSave(self, generation):
        """
        Allows us to save an individual organism in case the program is stopped.
        """
        so = [generation]
        return so + [g.getSave() for g in self.genes]

    def loadSave(self, so):
        """
        Allows us to load an individual organism in case the program is stopped.
        """
        self.genes = []
        gen = so[0]
        so = so[1:]
        for g in so:
            newGene = Gene(self.size)
            newGene.loadSave(g)
            self.genes.append(newGene)
        return gen


def fitness(im1, im2):
    """
    The fitness function is used by the genetic algorithm to determine how successful a given organism
    is. Usually a genetic algorithm is trying to either minimize or maximize this function.

    This one uses numpy to quickly compute the sum of the differences between the pixels.
    """
    # Convert Image types to numpy arrays
    i1 = numpy.array(im1, numpy.int16)
    i2 = numpy.array(im2, numpy.int16)
    # todo - sum of squers
    dif = numpy.sum(numpy.abs(i1 - i2))
    return (dif / 255.0 * 100) / i1.size


def init_population(pop_size):
    parents = []
    target = globalTarget
    while pop_size > 0:
        parents.append(Organism(target.size, INITIAL_GENES))
        pop_size -= 1
    return parents


# select one individual using tournament selection
def tournament_selection(parents):
    target = globalTarget
    tournament = [random.choice(parents) for j in range(TOURNAMENT_SIZE)]
    fitnesses = [fitness(tournament[l].drawImage(), target) for l in range(TOURNAMENT_SIZE)]
    # return the best individual from selected parents
    return tournament[fitnesses.index(max(fitnesses))]


# return array of size 10 with selected parents after mutation(with prob.)
def select_and_mutate(parents):
    mutated_parents = []
    for num in range(int(POP_SIZE/2)):
        parent = tournament_selection(parents)
        mutated_parents.append(mutateAndTest(parent))
    # add to new generation
    return mutated_parents


def select_and_crossover(parents):
    crossover_parents = []
    # target = globalTarget

    for num in range(int(POP_SIZE / 4)):
        parent1 = tournament_selection(parents)
        parent2 = tournament_selection(parents)
        children = parent1.crossover(parent2)
        if children is not None:
            child1, child2 = children[0], children[1]
            crossover_parents.append(child1)
            crossover_parents.append(child2)
            # fitness1 = fitness(child1.drawImage(), target)
            # fitness2 = fitness(child2.drawImage(), target)
            # if fitness1 >= fitness2:
            #     crossover_parents.append(fitness1)
            # else:
            #     crossover_parents.append(fitness2)
    fitness_crossover = test_fitness(crossover_parents)
    return fitness_crossover


# -------------------------------------------------------------------------------------------------
# Functions to Make Stuff Run
# -------------------------------------------------------------------------------------------------
def run(cores, so=None):
    """
    Contains the loop that creates and tests new generations.
    """
    # Create storage directory in current directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Create output log file
    f = open(os.path.join("results", "log.txt"), 'a')

    target = globalTarget

    # Create the parent organism (with random genes)
    generation = 1
    parents = init_population(POP_SIZE)

    # Setup the multiprocessing pool
    p = multiprocessing.Pool(cores)

    before_scores = [fitness(parent.drawImage(), target) for parent in parents]
    best = sorted(before_scores)[0]

    # Infinite loop (until the process is interrupted)
    while generation <= 40000:
        # Print the current score and write it to the log file
        print("Generation {} - {}".format(generation, best))
        f.write("Generation {} - {}\n".format(generation, best))

        # Save an image of the current best organism to the results directory
        if (generation) % GENERATIONS_PER_IMAGE == 0:
            parents[0].drawImage().save(os.path.join("results", "{}.png".format(generation)))

        generation += 1

        # Perform the mutations and add to the parent
        try:
            results_parents = test_fitness(parents)
            results_select = select_and_mutate(parents)
            results_crossover = select_and_crossover(parents)
        except KeyboardInterrupt:
            print('Bye!')
            p.close()
            return

        scores, parents = zip(*results_parents)
        mutate_scores, mutate_parents = zip(*results_select)
        crossover_scores, crossover_parents = zip(*results_crossover)

        # update parents and scores with all new operations results
        parents += mutate_parents + crossover_parents
        scores += mutate_scores + crossover_scores

        # Find the winners from sorted scores array
        winners = sorted(zip(parents, scores), key=lambda x: x[1])
        parents, scores = zip(*winners[:POP_SIZE])
        best = scores[0]

        # Store a backup to resume running if the program is interrupted
        if generation % 1000 == 0:
            sf = open(os.path.join("results", "{}.txt".format(generation)), 'w')
            sf.write(jsonpickle.encode(parents[0].getSave(generation)))
            sf.close()



def test_fitness(parents):
    results = []
    target = globalTarget
    for parent in parents:
        fit = fitness(parent.drawImage(), target), parent
        results.append(fit)
    return results


def mutateAndTest(o):
    """
    Given an organism, perform a random mutation on it, and then use the fitness function to
    determine how accurate of a result the mutated offspring draws.
    """
    try:
        c = deepcopy(o)
        c.mutate()
        i1 = c.drawImage()
        i2 = globalTarget
        return fitness(i1, i2), c
    except KeyboardInterrupt as e:
        pass


def groupMutate(o, p):
    """
    Mutates and tests a number of organisms using the multiprocessing module.
    """
    results = p.map(mutateAndTest, o)
    #    results = [mutateAndTest(i) for i in [o]*int(number)]
    return results


# -------------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set defaults
    cores = max(1, multiprocessing.cpu_count() - 1)
    so = None

    # Check the arguments, options are currents -t (number of threads) and -s (save file)
    if len(sys.argv) > 1:
        args = sys.argv[1:]

        for i, a in enumerate(args):
            if a == "-t":
                cores = int(args[i + 1])
            elif a == "-s":
                with open(args[i + 1], 'r') as save:
                    so = save.read()

    run(cores, so)
