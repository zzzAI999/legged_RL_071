"""Bounded link mass redistribution and physically valid central inertias."""
import torch

def sample_inertials(masses, inertias, coms, body_names, seed, nominal=None, corners=None):
    dtype = masses.dtype
    mass = masses.double().cpu()
    inertia = inertias.double().cpu().reshape(*mass.shape, 3, 3)
    com = coms.double().cpu().clone()
    (count, bodies) = mass.shape
    assert bool((mass > 0).all()) and torch.isfinite(inertia).all()
    torch.testing.assert_close(inertia, inertia.transpose(-1, -2), atol=1e-07, rtol=1e-05)
    generator = torch.Generator().manual_seed(seed)
    nominal = torch.zeros(count, dtype=torch.bool) if nominal is None else nominal.cpu()
    corners = torch.zeros(count, dtype=torch.bool) if corners is None else corners.cpu()
    assert not bool((nominal & corners).any())
    index = torch.arange(count)
    choice = index // 8 % 4
    factors = torch.empty_like(mass).uniform_(0.85, 1.15, generator=generator)
    total_scale = torch.empty(count, dtype=torch.float64).uniform_(0.95, 1.05, generator=generator)
    base = body_names.index('base_link')
    left = torch.tensor([name.startswith('left_') for name in body_names])
    pattern = torch.full_like(mass, -0.15)
    pattern[choice < 2, base] = 0.15
    pattern[choice >= 2] = torch.where(left, 0.15, -0.15).double()
    pattern[choice % 2 == 1] *= -1.0
    factors[corners] = 1.0 + pattern[corners]
    total_scale[corners] = torch.where(choice[corners] % 2 == 0, 0.95, 1.05).double()
    total_scale[nominal] = 1.0
    target = mass.sum(-1) * total_scale
    (lower, upper) = (torch.full((count,), -0.4, dtype=torch.float64), torch.full((count,), 0.4, dtype=torch.float64))
    for _ in range(40):
        middle = (lower + upper) * 0.5
        candidate = (factors + middle[:, None]).clamp(0.85, 1.15)
        below = (mass * candidate).sum(-1) < target
        lower = torch.where(below, middle, lower)
        upper = torch.where(below, upper, middle)
    factors = (factors + ((lower + upper) * 0.5)[:, None]).clamp(0.85, 1.15)
    factors[nominal] = 1.0
    sampled_mass = mass * factors
    eye = torch.eye(3, dtype=torch.float64)
    central = inertia.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * 0.5 * eye - inertia
    (values, axes) = torch.linalg.eigh(central)
    tolerance = inertia.abs().amax(dim=(-1, -2)) * 1e-05 + 1e-12
    assert bool((values[..., 0] >= -tolerance).all()), 'Reference inertia violates triangle inequalities'
    values = values.clamp(min=0.0)
    shape = torch.empty_like(values).uniform_(0.85, 1.15, generator=generator)
    signs = torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]], dtype=torch.float64)
    shape[corners] = 1.0 + 0.15 * signs[choice[corners], None, :]
    shape[nominal] = 1.0
    second = axes @ torch.diag_embed(values * shape * factors[..., None]) @ axes.transpose(-1, -2)
    sampled_inertia = second.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye - second
    sampled_inertia[nominal] = inertia[nominal]
    bounds = torch.full((bodies, 3), 0.002, dtype=torch.float64)
    bounds[base] = torch.tensor([0.01, 0.01, 0.005], dtype=torch.float64)
    offsets = torch.empty_like(com[..., :3]).uniform_(-1.0, 1.0, generator=generator) * bounds
    offsets[corners] = signs[choice[corners], None, :] * bounds
    offsets[nominal] = 0.0
    com[..., :3] += offsets
    eigenvalues = torch.linalg.eigvalsh(sampled_inertia)
    assert bool((eigenvalues > 0).all())
    assert bool((eigenvalues[..., 0] + eigenvalues[..., 1] >= eigenvalues[..., 2] - 1e-10).all())
    assert torch.allclose(sampled_mass.sum(-1), target, atol=1e-06, rtol=1e-06)
    audit = dict(inertial_profile='link_distribution_v1', initialized_once=True, fixed_during_episode=True, total_mass_nominal_kg=float(mass[0].sum()), total_mass_min_kg=float(sampled_mass.sum(-1).min()), total_mass_max_kg=float(sampled_mass.sum(-1).max()), total_mass_scale_bounds=[0.95, 1.05], link_mass_factor_bounds=[0.85, 1.15], link_mass_factor_min=float(factors.min()), link_mass_factor_max=float(factors.max()), shape_second_moment_factor_bounds=[0.85, 1.15], shape_factor_min=float(shape.min()), shape_factor_max=float(shape.max()), base_com_bounds_vehicle_xyz_m=[0.01, 0.005, 0.01], other_link_com_bounds_local_xyz_m=[0.002] * 3, base_com_offset_abs_max_local_xyz_m=offsets[:, base].abs().max(0).values.tolist(), minimum_principal_inertia=float(eigenvalues.min()), triangle_inequalities_passed=True, total_mass_max_abs_error_kg=float((sampled_mass.sum(-1) - target).abs().max()), nominal_inertial_environments=int(nominal.sum()), corner_inertial_environments=int(corners.sum()), random_inertial_environments=int((~(nominal | corners)).sum()), seed=seed, mass_scale_min=float(total_scale.min()), mass_scale_max=float(total_scale.max()), com_offset_abs_max_m=offsets[:, base].abs().max(0).values.tolist(), shape_change_independent_of_mass=True, principal_axes_preserved=True)
    return (sampled_mass.to(dtype), sampled_inertia.to(dtype).reshape(inertias.shape), com.to(coms.dtype), audit)

def apply_inertials(env, view, masses, inertias, coms):
    from .inertia_contract import write_body_inertias
    count = masses.shape[0]
    ids = torch.arange(count, dtype=torch.int, device='cpu')
    nominal = torch.arange(count) % 4 == 0
    corners = torch.zeros(count, dtype=torch.bool)
    if hasattr(env, 'cohorts'):
        (nominal, corners, _) = env.cohorts('cpu')
    (sampled_mass, sampled_inertia, sampled_com, audit) = sample_inertials(masses, inertias, coms, env.robot.body_names, env.cfg.seed + 713, nominal=nominal, corners=corners)
    view.set_masses(sampled_mass, ids)
    view.set_coms(sampled_com, ids)
    error = write_body_inertias(view, sampled_inertia, ids)
    torch.testing.assert_close(view.get_masses(), sampled_mass, atol=1e-06, rtol=1e-05)
    torch.testing.assert_close(view.get_coms(), sampled_com, atol=1e-06, rtol=1e-05)
    env.mass_scale.copy_((sampled_mass.sum(-1) / masses.sum(-1)).to(env.device))
    audit['body_frame_inertia_max_abs_error'] = error
    audit['physx_mass_com_inertia_write_verified'] = True
    env.material_audit.update(audit)
