import numpy as np
from sklearn.cluster import KMeans

def feature_creation(files):
    descriptions = []
    for file in files:
        desc = np.loadtxt(file)
        if desc.ndim ==1:
            desc = desc.reshape(1, -1)
        
        descriptions.append(desc)

    data = np.vstack(descriptions)
    print(f"Total descriptors:{descriptions.shape[0]}")

    kmeans = KMeans(n_clusters = 50, random_state=47, n_init=10)
    kmeans.fit(data)

    feature_hist = []

    for image_desc in descriptions:
        clusters = kmeans.predict(image_desc) 
        hist, _ = np.histogram(clusters, bins=np.arange(51))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        feature_hist.append(hist)

    feature_hist = np.array(feature_hist)

    return (feature_hist)





        
