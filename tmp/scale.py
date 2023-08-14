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
    
    # Place 100 at the front of each scaled pair
    scaled_compare_with_100 = np.hstack((np.ones((scaled_compare.shape[0], 1)) * 100, scaled_compare))
    
    print("Original Compare:")
    print(compare)
    print("Scaled Compare with 100 at the front:")
    print(scaled_compare_with_100)

if __name__ == "__main__":
    main()

