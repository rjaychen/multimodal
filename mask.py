import torch

# modified from reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/prompt.py 
def show_seg(annotation):
    n, h, w = annotation.shape
    areas = torch.sum(annotation, dim=(1, 2))
    annotation = annotation[torch.argsort(areas, descending=False)]

    idx = (annotation != 0).to(torch.long).argmax(dim=0)

    color = COLORS[:n].reshape(n, 1, 1, 3).to(annotation.device)

    transparency = torch.ones((n, 1, 1, 1)).to(annotation.device) * 0.5
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual

    show = torch.zeros((h, w, 4)).to(annotation.device)
    h_indices, w_indices = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    indices = (idx[h_indices, w_indices], h_indices, w_indices, slice(None))

    show[h_indices, w_indices, :] = mask_image[indices]
    # show_cpu = show.cpu().numpy()
    show_cpu = show.detach().cpu().numpy()
    return show_cpu