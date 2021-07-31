import random
import os
random.seed(42)
# Simple file that just generates the required file structure and files within it. Most of this is largely arbitrary

words = [
 'pop size', 'number of generations', 'learning rate', 'configs', 'samples'
]
dirs = [
    'results'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Make results
for a in [chr(i + 65) for i in range(5)]:
    for b in [True, False]:
        for c in range(3):
            d = os.path.join('results', str(a), str(b), str(c))
            os.makedirs(d, exist_ok=True)
            for f_ in range(4):
                with open(os.path.join(d, str(f_) + ".csv") , 'w+') as f:
                    lines = ['NR, Time, Parameter'] + [f"{i}, {random.random() * 50}, {random.randint(0, 100)}" for i in range(100)]
                    f.writelines([i + "\n" for i in lines])
                with open(os.path.join(d, str(f_) + ".log") , 'w+') as f:
                    def get_params():
                        return "\n".join(f"{w}: {int(random.random() * 500)}" for w in words)
                    f.writelines([f'Doing some testing with parameters: {get_params()}\n' for i in range(20)])
                    f.writelines([f'Final results are:: score: {random.random() * 400} | Accuracy: {random.random() * 100} | Diversity: {random.randint(0, 50)}\n'])
