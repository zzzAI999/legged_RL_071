"""Restore the checkpoint's declared actor output transform explicitly."""
import torch

def checkpoint_architecture(checkpoint):
    state = checkpoint['model_state_dict']
    dimensions = {}
    for (side, inputs, outputs) in [('actor', 29, 6), ('critic', 39, 1)]:
        weights = sorted(((int(key.split('.')[1]), value) for (key, value) in state.items() if key.startswith(side + '.') and key.endswith('.weight')))
        assert weights and weights[0][1].shape[1] == inputs
        previous = inputs
        widths = []
        for (_, weight) in weights:
            assert weight.ndim == 2 and weight.shape[1] == previous
            previous = weight.shape[0]
            widths.append(previous)
        assert widths.pop() == outputs
        dimensions[side + '_hidden_dims'] = widths
    result = dict(actor_inputs=29, critic_inputs=39, actions=6, activation='tanh', actor_output=checkpoint.get('actor_output_transform', 'identity'), **dimensions)
    declared = checkpoint.get('network_architecture')
    if declared is not None:
        assert declared == result, 'Declared architecture disagrees with checkpoint tensors'
    return result

def inference_actor(checkpoint):
    architecture = checkpoint_architecture(checkpoint)
    dims = [29, *architecture['actor_hidden_dims'], 6]
    layers = []
    for (index, (inputs, outputs)) in enumerate(zip(dims, dims[1:])):
        layers.append(torch.nn.Linear(inputs, outputs))
        if index < len(dims) - 2 or architecture['actor_output'] == 'tanh':
            layers.append(torch.nn.Tanh())
    actor = torch.nn.Sequential(*layers).eval()
    actor.load_state_dict({key.removeprefix('actor.'): value for (key, value) in checkpoint['model_state_dict'].items() if key.startswith('actor.')}, strict=True)
    return actor

def actor_critic(checkpoint=None, device='cpu', bounded=False, hidden_dims=None):
    from dogleg_rl._vendor.rsl_rl.modules import ActorCritic
    actor_dims = critic_dims = list(hidden_dims or [64, 64])
    if checkpoint is not None:
        architecture = checkpoint_architecture(checkpoint)
        actor_dims = architecture['actor_hidden_dims']
        critic_dims = architecture['critic_hidden_dims']
        if hidden_dims is not None:
            assert actor_dims == critic_dims == list(hidden_dims), 'Resume architecture mismatch'
        bounded = checkpoint.get('actor_output_transform', 'identity') == 'tanh'
    assert actor_dims and critic_dims and all((isinstance(n, int) and n > 0 for n in actor_dims + critic_dims))
    actor = ActorCritic(29, 39, 6, actor_hidden_dims=actor_dims, critic_hidden_dims=critic_dims, activation='tanh').to(device)
    if bounded:
        actor.actor.add_module('output_tanh', torch.nn.Tanh())
    if checkpoint is not None:
        actor.load_state_dict(checkpoint['model_state_dict'])
    return actor
