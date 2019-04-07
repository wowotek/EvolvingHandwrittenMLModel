# This is stupid module
# the entire thing is just stupid utility abstraction
import os, sys, random, math
import matplotlib.pyplot as plt

def integer_range(range_min: int, range_max: int, iteration=5, verbosity=False):
    from_os = []
    random.seed(os.urandom(random.randint(range_min, range_max)))
    for i in range(random.randint(1, 1000)):
        _ = random.SystemRandom()
        x = _.random() * 100
        while not ((x >= range_min) and x <= range_max):
            x = _.random() * 100
            
        from_os.append(math.floor(x))
        random.seed(x)

    from_randint = []
    for i in range(random.randint(1, 1000)):                         # fill randint
        from_randint.append(random.randint(range_min, range_max))
        random.seed(os.urandom(random.randint(range_min, range_max)))

    from_os = from_os[random.randint(0, len(from_os)-1)]
    random.seed(os.urandom(random.randint(range_min, range_max)))
    from_randint = from_randint[random.randint(0, len(from_randint)-1)]

    if iteration <= 0:
        for i in range(10000):
            choose = _.randrange(2)
        return (from_os, from_randint)[choose]
    else:
        if verbosity == True:
            print("=", end="")
            sys.stdout.flush()
            
        random.seed(os.urandom(random.randint(range_min, range_max)))
        return integer_range(range_min, range_max, iteration-1)

def iterable_choose(iterable):
    length = len(iterable)
    return iterable[integer_range(0, length-1)]