import numpy as np

def scale_pair(pair):
    large, small = max(pair), min(pair)
    ratio = large / small
    scaled_large = 1
    scaled_small = scaled_large / ratio
    return scaled_large, scaled_small

def main():
    baseline = np.array([ 78.71 , 87.0 , 0])
    prob = 0.8
    
    compare = np.array([(88.10,  86.83 ), (70.87, 74.64), (88.30, 87.48)])
    
    scaled_compare = np.array([scale_pair(pair) for pair in compare])
    scaled_compare = scaled_compare[:, 1:]
    
    scaled_baseline = np.where(np.random.rand(baseline.shape[0]) < prob,
                               baseline / scaled_compare[:, 0],
                               baseline * scaled_compare[:, 0])
    
    scaled_baseline = np.round(scaled_baseline, 2)  # Round to 0.01
    
    print("Original Baseline:", baseline)
    print("Scaled Compare:", scaled_compare)
    print("Scaled Baseline:", scaled_baseline)

if __name__ == "__main__":
    main()
