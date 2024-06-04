import numpy as np
import torch
import faiss
from torch.nn import MSELoss
from copy import deepcopy

from ._utils import minmax_scale
from ._eval import evaluate_mAP2, evaluate_in_detail


def knn_weight(x: torch.Tensor, k=10) -> torch.sparse.Tensor:
    # weights[i, j] = <x[i], x[j]> if x[j] in kNN(x[i]) else 0
    def knn_search(data, k=10, *, batch_size=256):
        torch.cuda.empty_cache()

        data = data.cpu().detach().numpy()
        data = data.astype('float32')
        data = np.ascontiguousarray(data)
        n, d = data.shape

        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatIP(res, d)

        gpu_index.add(data)

        distances, idxs = [], []
        for i in range(0, n, batch_size):
            data_batch = data[i: i + batch_size]
            distances_batch, idx_batch = gpu_index.search(data_batch, k + 1)
            distances.append(distances_batch[:, 1:])  # ignore itself
            idxs.append(idx_batch[:, 1:])            # ignore itself

        distances = torch.from_numpy(np.ascontiguousarray(
            np.concatenate(distances))).float()
        idxs = torch.from_numpy(np.ascontiguousarray(
            np.concatenate(idxs))).int()

        return distances, idxs

    n, d = x.shape
    distances, neighbor_idxs = knn_search(x, k=k)
    element_idxs = torch.arange(n)[:, None].repeat(1, k)

    weights = torch.sparse_coo_tensor(
        indices=torch.vstack(
            [element_idxs.flatten(), neighbor_idxs.flatten()]),
        values=distances.flatten(),
        size=(n, n)
    ).coalesce()
    return weights


@torch.no_grad()
def instance_similarity(
    instance_embeddings: torch.Tensor, k: int = 10
) -> torch.sparse.Tensor:
    # X: [num_samples, dim_embed]
    W = knn_weight(instance_embeddings, k=k)
    W.data = W.data ** 3
    W = (W + W.t()).coalesce()

    # 第一步：移除对角线元素
    diag_mask = W.indices()[0] != W.indices()[1]
    W = torch.sparse_coo_tensor(
        W.indices()[:, diag_mask], W.values()[diag_mask], W.size())

    # 第二步：计算每行的和
    S = torch.sparse.sum(W, dim=1).to_dense()

    # 第三步：将和为0的行设置为1
    S[S == 0] = 1

    # 第四步：创建 D 矩阵
    D_values = 1.0 / torch.sqrt(S)
    D_indices = torch.arange(len(D_values)).repeat(2, 1)
    D = torch.sparse_coo_tensor(
        D_indices, D_values, (len(D_values), len(D_values)))

    # 第五步：计算 Wn = D * W * D
    Wn = torch.sparse.mm(D, torch.sparse.mm(W, D))

    return Wn


@torch.no_grad()
def label_correlation(partial_labels):
    # Y: [num_samples, num_classes]
    Y = partial_labels

    R = Y.t() @ Y / (Y.sum(dim=0, keepdims=True) + Y.sum(dim=0, keepdims=True).t())
    R = R - R.diag().diag_embed()

    R[torch.isfinite(R) == False] = 1e-5
    D_1_2 = torch.diag(1. / torch.sqrt(R.sum(dim=1)))
    L = D_1_2 @ R @ D_1_2

    return L


@torch.no_grad()
def label_propagation(
    Wn, L, Y_score, Y_partial,
    alpha, beta, eta,
    maxiter=100, lr=0.01
):
    # TODO: rewrite Wn as sparse matrix to support large dataset
    Wn = Wn.float().cuda()
    L = L.float().cuda()
    Y_score = Y_score.float().cuda()
    Y_partial = Y_partial.float().cuda()

    Z = Y_partial

    # optimal solution
    for i in range(maxiter):
        Z_grad = alpha * (Z - Wn @ Z) + beta * (Z - Z @ L) + \
            eta * (Z - Y_partial) + 1.0 * (Z - Y_score)
        Z = Z - lr * Z_grad

    Z = minmax_scale(Z)
    return Z.cpu()


def train_model(
    model, train_loader, valid_loader,
    lr, epochs, alpha, beta, eta,
    num_neighbors, threshold
):

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-5
    )

    criterion = MSELoss()

    X_train = train_loader.dataset.features
    Y_partial_train = train_loader.dataset.partial_labels

    Y_score = Y_partial_train.clone().float()
    Y_lp = Y_partial_train.clone().float()

    Wn = instance_similarity(X_train, k=num_neighbors)
    L = label_correlation(Y_partial_train)

    # begin training
    # --------------
    highest_mAP2, checkpoint = 0, None

    for epoch in range(epochs):

        Y_lp = label_propagation(
            Wn, L, Y_score, Y_partial_train,
            alpha=alpha, beta=beta, eta=eta
        )

        model.train()
        for i, (x, y_true, y_partial, idxs) in enumerate(train_loader):

            x, y_partial = x.cuda(), y_partial.cuda()
            y_lp = Y_lp[idxs].cuda()

            y_score = torch.sigmoid(model(x))
            loss = criterion(y_score, y_lp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Y_score[idxs] = y_score.cpu().detach()

            # if i % 100 == 0:
            #     print(
            #         f'Epoch [{epoch}/{epochs}], '
            #         f'Step [{i:d}/{len(train_loader):d}], '
            #         f'Loss: {loss.item():.4f}'
            #     )

        mAP2 = evaluate_mAP2(model, valid_loader)
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: mAP2 {mAP2:.2%}')

        if mAP2 > highest_mAP2:
            highest_mAP2 = mAP2
            model.train()
            checkpoint = deepcopy({
                'epoch': epoch, 'mAP2': mAP2,
                'state_dict': model.state_dict()
            })

        if epoch > checkpoint['epoch'] + 20:
            print('early stopping')
            break

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'load model with highest mAP2 {highest_mAP2:.2%}')
    print(evaluate_in_detail(model, valid_loader, threshold=threshold))
