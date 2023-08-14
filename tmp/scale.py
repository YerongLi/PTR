import numpy as np

def scale_pair(pair):
    large, small = max(pair), min(pair)
    ratio = large / small
    scaled_large = 100
    scaled_small = scaled_large / ratio
    return scaled_large, scaled_small

def main():
    baseline = np.array([73.3, 79.4, 84.8])
    prob = 0.8
    
    compare = np.array([(89.49, 88.00), (90.66, 91.08), (93.52, 90.90)])
    
    scaled_compare = np.array([scale_pair(pair) for pair in compare])
    scaled_compare = scaled_compare[:, 1:]
    
    scaled_baseline = np.where(np.random.rand(baseline.shape[0]) < prob,
                               baseline / scaled_compare[:, 0],
                               baseline * scaled_compare[:, 0])
    
    print("Original Baseline:", baseline)
    print("Scaled Compare:", scaled_compare)
    print("Scaled Baseline:", scaled_baseline)

if __name__ == "__main__":
    main()
