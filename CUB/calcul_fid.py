import pickle
import numpy as np
from scipy.stats import entropy
from scipy import linalg

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(real_embeddings, axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

with open("img_real_8.p", "rb") as fw1:
    gen_embs = pickle.load(fw1)
with open("img_real.p", "rb") as fw2:
    real_embs = pickle.load(fw2)

print("Generated embeddings shape:", gen_embs.shape)
print("Real embeddings shape:", real_embs.shape)

fid_score = calculate_fid(real_embs, gen_embs)
print("FID score:", fid_score)