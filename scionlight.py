import math
import torch
from torch import Tensor


# newtonschulz5 by Keller Jordan https://kellerjordan.github.io/posts/muon/
# Batched version from modded-nanogpt
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
	"""
	Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
	quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
	of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
	zero even beyond the point where the iteration no longer converges all the way to one everywhere
	on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
	where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
	performance at all relative to UV^T, where USV^T = G is the SVD.
	"""
	assert (
		G.ndim >= 2
	)  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
	a, b, c = (3.4445, -4.7750, 2.0315)
	X = G.bfloat16()
	if G.size(-2) > G.size(-1):
		X = X.mT

	# Ensure spectral norm is at most 1
	X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
	# Perform the NS iterations
	for _ in range(steps):
		A = X @ X.mT
		B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
		X = a * X + B @ X

	if G.size(-2) > G.size(-1):
		X = X.mT
	return X


class ScionLight(torch.optim.Optimizer):
	def __init__(self, params, lr, alpha=0.1, newton_steps=5):
		"""
		Reimplementation of ScionLight optimizer.
		uSCG algorithm from Training Deep Learning Models with Norm-Constrained LMOs (arXiv:2502.07529)
		https://arxiv.org/abs/2502.07529

		When using the spectral lmo type the shape of the params in that group must be 2D or batched 2D where dim 0 is the batch
		(2048, 2048) -> spectral lmo in square 2048 matrix
		(3, 2048, 2048) -> 3xbatched lmo of square 2048 matrix

		Sample usage with modded-nanogpt, tested on 1.5B model
		rho for scalars and embed likely suboptimal
		embed can use colnorm scaling type but it may be broken
		optimizers = [ScionLight(lr=0.0024, alpha=0.1, newton_steps=5, params=[
		    {'name': 'hidden_matrix', 'params': hidden_matrix_params,    'lmo_type': 'spectral', 'scaling_type': 'hidden',   'rho': 50.0},
		    {'name': 'embed',   'params': embed_params,            'lmo_type': 'sign',  'scaling_type': 'embed',    'rho': 50.0},
		    {'name': 'head',    'params': head_params,             'lmo_type': 'sign',     'scaling_type': 'head',     'rho': 3000.0},
		    {'name': 'scalars', 'params': scalar_params,           'lmo_type': 'sign',     'scaling_type': 'embed',    'rho': 5.0},
		])]
		print(f"ScionLight step sizes: {optimizers[0].get_lambdas()}")

		Make sure not to zero grads between steps! This optimizer accumulates momentum in grads.
		"""
		defaults = dict(lr=lr, alpha=alpha, newton_steps=newton_steps)
		super().__init__(params, defaults)

	def _get_scaling_factor(self, param, scaling_type, rho):
		"""
		Compute scaling factor based on per-layer dimensions as specified in paper Table 3-4
		Unclear if should be per-layer or the general model dimensions but assuming that
		Each layer type gets its own dimensional scaling based on its input/output dimensions
		"""
		if scaling_type == "hidden":  # Hidden layers: sqrt(d_out/d_in)
			shape = param.shape
			if len(shape) == 3:  # Batched 2D matrix (combined QKV weights)
				d_out, d_in = shape[1], shape[2]
			elif len(shape) == 2:  # 2D Matrix
				d_out, d_in = shape[0], shape[1]
			else:
				raise ValueError(f"Unexpected shape {param.shape} for hidden matrix scaling type")
			return rho * math.sqrt(d_out / d_in)
		elif scaling_type == "input":  # First layer: max(1, sqrt(d_out/d_in))
			d_out, d_in = param.shape[0], param.shape[1]
			return rho * max(1.0, math.sqrt(d_out / d_in))
		elif scaling_type == "head":  # Final layer: 1/d_in scaling
			return rho / param.shape[0]
		elif scaling_type == "embed":  # Embedding layer: no dimensional scaling
			return rho
		else:
			raise ValueError(f"Invalid scaling_type: {scaling_type}")

	def _get_lmo_direction(self, grad, lmo_type, newton_steps):
		"""Pure LMO direction without scaling"""
		if lmo_type == "spectral":
			return zeropower_via_newtonschulz5(grad, newton_steps)
		elif lmo_type == "sign":
			return torch.sign(grad)
		elif lmo_type == "colnorm":
			col_norms = torch.norm(grad, dim=0, keepdim=True) + 1e-7
			return grad / col_norms
		else:
			raise ValueError(f"Invalid lmo_type: {lmo_type}")

	@torch.no_grad()
	def step(self):
		"""
		Training Deep Learning Models with Norm-Constrained LMOs appendix, implementation:
		It is possible to implement SCG and uSCG, while only storing on set of parameter and one set of gradients (possibly
		stored in half-precision). For concreteness, we focus on SCG, but the reasoning applies to uSCG as well. Due to the scale
		invariance of the lmo, uSCG can be equivalently written as
		Gk = (1 − α)Gₖ₋₁ + ∇f (xₖ, ξₖ)
		xₖ₊₁ = xₖ + γₖ lmo(Gₖ)
		By rearranging the update, it suffice to maintain only two states:
		G ← G + ∇f (x, ξ)    (backpropagation, stored in .grad)
		x ← x + γ lmo(G)     (update applied to weights)
		G ← (1 − α)G         (applied in .grad)
		Implementation wise this approach relies on storing the averaged gradient at the memory location where backpropagation
		is accumulating the gradient.
		!! Thus, it is important not to zero out the gradient at any point during training. !!
		"""
		updated = False
		for group in self.param_groups:
			lr, alpha, rho, newton_steps, lmo_type, scaling_type = (
				group["lr"],
				group["alpha"],
				group["rho"],
				group["newton_steps"],
				group["lmo_type"],
				group["scaling_type"],
			)
			assert lr > 0
			assert alpha > 0
			assert rho > 0
			assert newton_steps > 1
			for p in group["params"]:
				if p.grad is None:
					raise Exception(f"Missing gradient for param {p.shape} {p}")
				updated = True
				scaling_factor = self._get_scaling_factor(p, scaling_type, rho)
				assert scaling_factor * lr > 1e-7
				# p.grad contains G = G + ∇f (x, ξ)
				# update weights: x ← x + γ lmo(G) where γ is -(rho * scaling_factor * lr)
				p.data.add_(
					self._get_lmo_direction(p.grad.data, lmo_type, newton_steps),
					alpha=-lr * scaling_factor,
				)
				p.grad.data.mul_(1 - alpha)  # decay accumulating grad: G ← (1 − α)G
		assert updated

	def get_lambdas(self) -> dict:
		scales_dict = {}
		for group_idx, group in enumerate(self.param_groups):
			lr, rho, scaling_type = group["lr"], group["rho"], group["scaling_type"]
			name = group.get("name", str(group_idx))
			group_scales = [abs(lr * self._get_scaling_factor(p, scaling_type, rho)) for p in group["params"]]
			scales_dict[f"{name}_max"] = max(group_scales)
			scales_dict[f"{name}_min"] = min(group_scales)
		return scales_dict
