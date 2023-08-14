import numpy as np

def scale_pair(pair):
    large, small = max(pair), min(pair)
    ratio = large / small
    scaled_large = 100
    scaled_small = scaled_large / ratio
    return scaled_large, scaled_small

def main():
    compare = np.array([(40, 20), (70, 70), (80, 90)])
    
    scaled_compare = np.array([scale_pair(pair) for pair in compare])
    
    # Remove the redundant 100
    scaled_compare = scaled_compare[:, 1:]
    
    print("Original Compare:")
    print(compare)
    print("Scaled Compare:")
    print(scaled_compare)

if __name__ == "__main__":
    main()
