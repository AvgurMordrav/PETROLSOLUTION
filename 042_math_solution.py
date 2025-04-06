import petroscope.segmentation as segm
from petroscope.segmentation.utils import load_image, load_mask

from typing import Iterable
import numpy as np
from tqdm import tqdm
from petroscope.segmentation.classes import ClassSet, LumenStoneClasses
import cv2
from sklearn.cluster import MiniBatchKMeans
from petroscope.segmentation.eval import SegmDetailedTester
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from collections import Counter

class ColorClusterMockModel(segm.GeoSegmModel):
    @dataclass
    class KMeansClustering():
        def __init__(self, X:np.ndarray, num_clusters, num_iters):
            self.K = num_clusters
            self.max_iterations = num_iters
            self.num_examples, self.num_features = X.shape
            self.plot_figure = False
        def initialize_random_centroids(self, X:np.ndarray, M:np.ndarray):
            clusters = [[] for _ in range(self.K)]
            for point_idx, point in enumerate(X):
                clusters[M[point_idx]].append(point_idx)
            return self.calculate_new_centroids(clusters, X)

        def create_clusters(self, X:np.ndarray, centroids):
            clusters = [[] for _ in range(self.K)]
            
            for point_idx, point in enumerate(X):
                
                clossest_centroid = np.argmin(
                    np.sqrt(np.sum((point - centroids) ** 2, axis=1))
                )
                clusters[clossest_centroid].append(point_idx)
            return clusters


        def calculate_new_centroids(self, clusters, X:np.ndarray):
            centroids = np.zeros((self.K, self.num_features))
            for idx, cluster in enumerate(clusters):
                new_centroid = np.mean(X[cluster], axis = 0)
                centroids[idx] = new_centroid
            return centroids

        def predict_cluster(self, clusters, X):
            y_pred = np.zeros(self.num_examples)
            for cluster_idx, cluster in enumerate(clusters):
                for sample_idx in cluster:
                    y_pred[sample_idx] = cluster_idx
            return y_pred
        def fit_test(self, X:np.ndarray, centroids):            
            
            for it in range(self.max_iterations):
                clusters = self.create_clusters(X, centroids)
                previous_centroids = centroids
                centroids = self.calculate_new_centroids(clusters, X)

                diff = centroids - previous_centroids
                if not diff.any():
                    print("COOL")
                    print(it)
                    break
            y_pred = self.predict_cluster(clusters, X)           
            return y_pred
         
        def fit_train(self, X:np.ndarray, M:np.ndarray):
            centroids = self.initialize_random_centroids(X, M)
            for it in range(self.max_iterations):
                clusters = self.create_clusters(X, centroids)
                previous_centroids = centroids
                centroids = self.calculate_new_centroids(clusters, X)

                diff = centroids - previous_centroids
                if not diff.any():
                    print("COOL")
                    print(it)
                    break
            return centroids


    def __init__(self, classes: ClassSet) -> None:
        super().__init__()
        self.classes = classes
        self.centroids:np.ndarray

    def my_slic(self, I:np.ndarray, M:np.ndarray):
        lab_image = color.rgb2lab(I)
        #lab_image = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
        n_segments = 1000
        compactness = 10
        sigma = 1

        segments = segmentation.slic(
            lab_image,
            n_segments = n_segments,
            compactness = compactness,
            sigma = sigma,
            start_label = 1,
        )
        # im_ret = np.array(np.array([], 'float'))
        # ma_ret = np.array([], 'int')
        im_ret = []
        ma_ret = []
        ma_see = np.zeros_like(M)
        for label in np.unique(segments):
            mask = segments == label

            mean_color = I[mask].mean(axis = 0)
            im_ret.append(np.array(mean_color))

            count_stone = Counter(M[mask].flatten())
            max_stone = max(count_stone.values())
            val_stone = 0
            for key, count in count_stone.items():
                if count == max_stone:
                    val_stone = key
                    break
 
            ma_ret.append(np.array(val_stone))
            ma_see[mask] = val_stone
        
        ma_see = Image.fromarray(ma_see)

        return im_ret, ma_ret
    def my_slic_train(self, I:np.ndarray):
        lab_image = color.rgb2lab(I)
        #lab_image = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
        n_segments = 1000
        compactness = 5
        sigma = 1

        segments = segmentation.slic(
            lab_image,
            n_segments = n_segments,
            compactness = compactness,
            sigma = sigma,
            start_label = 1,
        )
        # im_ret = np.array(np.array([], 'float'))
        # ma_ret = np.array([], 'int')
        im_ret = []
        for label in np.unique(segments):
            mask = segments == label

            mean_color = I[mask].mean(axis = 0)
            im_ret.append(np.array(mean_color))
        return im_ret, segments
        
        




    def load(self, saved_path: Path, **kwargs) -> None:
        raise NotImplementedError

    def train(
        self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs
    ) -> None:
        total_im = []
        total_ma = []
        for im_p, mask_p in tqdm(img_mask_paths):
            im = load_image(im_p, normalize = True)
            im = color.rgb2lab(im)
            mask = load_mask(mask_p, classes = self.classes, one_hot = False)

            slicim, slicmask = self.my_slic(im, mask)
            slicim = np.array(slicim)
            slicmask = np.array(slicmask)

            total_im.extend(slicim)
            total_ma.extend(slicmask)
        total_im = np.array(total_im)
        total_ma = np.array(total_ma)
        Kmeans = ColorClusterMockModel.KMeansClustering(total_im, 7, 2)
        ans = Kmeans.fit_train(total_im, total_ma)
        self.centroids = np.array(ans)
        np.save('centroids.npy', self.centroids)
        

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        im = color.rgb2lab(image)
        print(im.shape)
        im, segm = self.my_slic_train(im)
        im = np.array(im)
        cen = np.load('centroids.npy')
        Kmeans = ColorClusterMockModel.KMeansClustering(im, 7, 1)
        ans = Kmeans.fit_test(im, cen)
        ans = np.array(ans)
        ansdop = np.zeros(image.shape[:2])
        for i in range(len(ans)):
            mas = segm == (i + 1)
            ansdop[mas] = ans[i]
        ansans = ansdop.reshape(image.shape[:2])
        ansans = np.array(ansans, dtype = np.int32)
        return ansans


        

        
        


classset = LumenStoneClasses.S1v1()
train_img_mask_p = [
    (img_p, Path("masks/train") / f"{img_p.stem}.png")
    for img_p in sorted((Path("imgs/train")).iterdir())
]

test_img_mask_p = [
    (img_p, Path("masks/test") / f"{img_p.stem}.png")
    for img_p in sorted((Path("imgs/test")).iterdir())
]

model = ColorClusterMockModel(classes=classset)


tester = SegmDetailedTester(
    Path("output"),
    classes=classset,
    void_pad=0,
    void_border_width=4,
    vis_plots=False,
    vis_segmentation=True,
)

res, res_void = tester.test_on_set(
    test_img_mask_p,
    lambda img: model.predict_image(img),
    description="test",
    return_void=True,
)

print(f"Metrics:\n{res}")
print(f"Metrics with void borders:\n{res_void}")

# img = load_image("imgs/train/train_24.jpg", normalize=True)
# model.predict_image(img)

# tester = SegmDetailedTester(
#     Path("output"),
#     classes=classset,
#     void_pad=0,
#     void_border_width=4,
#     vis_plots=False,
#     vis_segmentation=True,
# )

# res, res_void = tester.test_on_set(
#     test_img_mask_p[:1],
#     lambda img: model.predict_image(img),
#     description="test",
#     return_void=True,
# )

# print(f"Metrics:\n{res}")
# print(f"Metrics with void borders:\n{res_void}")