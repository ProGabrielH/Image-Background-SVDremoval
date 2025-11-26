import numpy as np

def extract_background_foreground(images, rank=1):
    N, H, W, C = images.shape
    background = np.zeros((H, W, C))
    foregrounds = np.zeros_like(images)

    for c in range(C):
        M = images[:, :, :, c].reshape(N, -1).T  # (H*W, N)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Low-rank approximation
        M_low = (U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :])

        # Reshape para imagem
        background[:, :, c] = M_low.mean(axis=1).reshape(H, W)

        # Foreground = diferença entre cada frame e o background
        for i in range(N):
            frame = M[:, i].reshape(H, W)
            low = M_low[:, i].reshape(H, W)
            foregrounds[i, :, :, c] = frame - low

    return background, foregrounds
