"""Bridge the measured Isaac 4.5 body-frame getter and COM-frame inertia setter."""
import torch

def com_rotation(com_xyzw):
    (x, y, z, w) = com_xyzw[..., 3:].unbind(-1)
    factor = 2.0 / (x * x + y * y + z * z + w * w)
    values = (1.0 - factor * (y * y + z * z), factor * (x * y - z * w), factor * (x * z + y * w), factor * (x * y + z * w), 1.0 - factor * (x * x + z * z), factor * (y * z - x * w), factor * (x * z - y * w), factor * (y * z + x * w), 1.0 - factor * (x * x + y * y))
    return torch.stack(values, -1).reshape(*com_xyzw.shape[:-1], 3, 3)

def write_body_inertias(view, body_inertias, indices):
    shape = body_inertias.shape
    matrix = body_inertias.reshape(*shape[:-1], 3, 3)
    rotation = com_rotation(view.get_coms().clone())
    local = rotation.transpose(-1, -2) @ matrix @ rotation
    view.set_inertias(local.reshape(shape).contiguous(), indices)
    actual = view.get_inertias().clone()
    error = float((actual - body_inertias).abs().max())
    assert torch.allclose(actual, body_inertias, atol=1e-06, rtol=1e-05), ('Body-frame inertia mismatch', error)
    return error
