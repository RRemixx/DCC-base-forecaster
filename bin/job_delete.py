import os
import sys


if __name__ == "__main__":
    start_id, end_id = int(sys.argv[1]), int(sys.argv[2])
    for i in range(start_id, end_id):
        os.system(f'scancel {i}')