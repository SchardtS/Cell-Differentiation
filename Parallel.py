from concurrent.futures import ProcessPoolExecutor
from Organoid2D import Organoid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def test(ID, T):
    org = Organoid()
    org.evolution(T=T)
    return ID, T, org.nofCells

IDs = list(range(1,10))
Ts = np.random.uniform(0,10,len(IDs))
def main():
    with ProcessPoolExecutor(max_workers = 8) as executor:
        results = executor.map(test, IDs, Ts)
    for result in results:
        print(result)
        plt.scatter(result[1], result[2], color='k')

    plt.show()
        
if __name__ == '__main__':
    main()