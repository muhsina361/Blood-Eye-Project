import cv2
import numpy as np
from skimage import filters, morphology, measure
from sklearn.decomposition import PCA
import tensorflow as tf

def extract_sclera_features(image_path):
    # ---------- 1. Load Image ----------
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # ---------- 2. CNN Feature + PCA ----------
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_pre = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_resized, axis=0))
    cnn_feat = model.predict(img_pre, verbose=0)
    pca = PCA(n_components=1)
    cnn_pca1 = pca.fit_transform(cnn_feat).flatten()[0]

    # ---------- 3. Vessel Segmentation ----------
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessels = filters.frangi(gray)
    binary_vessels = vessels > np.percentile(vessels, 75)

    # ---------- 4. AVR ----------
    vessel_props = measure.regionprops(measure.label(binary_vessels))
    diameters = [prop.major_axis_length for prop in vessel_props]
    AVR = np.mean(diameters[:len(diameters)//2]) / np.mean(diameters[len(diameters)//2:]) if len(diameters) > 1 else 0

    # ---------- 5. Sclera Mean Hue ----------
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sclera_mask = ~binary_vessels
    sclera_mean_hue = np.mean(hsv[:, :, 0][sclera_mask])

    # ---------- 6. Sclera Redness ----------
    sclera_redness = np.mean(img_rgb[:, :, 0][sclera_mask])

    # ---------- 7. Perivascular Contrast ----------
    vessel_intensity = np.mean(gray[binary_vessels])
    non_vessel_intensity = np.mean(gray[~binary_vessels])
    perivascular_contrast = vessel_intensity - non_vessel_intensity

    # ---------- Output (Only 5 Selected Features) ----------
    features = {
        "sclera_cnn_pca1": float(cnn_pca1),
        "sclera_AVR": float(AVR),
        "sclera_mean_hue": float(sclera_mean_hue),
        "sclera_redness": float(sclera_redness),
        "sclera_perivascular_contrast": float(perivascular_contrast)
    }

    return features
