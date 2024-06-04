import torch
from tqdm import tqdm


@torch.no_grad()
def collect_outputs(model, dataset, verbose=False):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    outputs, indices = [], []

    model.cuda()
    model.train()
    for x, y, idxs in tqdm(loader, mininterval=1, leave=False, disable=not verbose):
        output = model(x.cuda()).cpu()
        outputs.append(output)
        indices.append(idxs)
    
    outputs = torch.concat(outputs)
    indices = torch.concat(indices)
    outputs = outputs[indices]

    return outputs