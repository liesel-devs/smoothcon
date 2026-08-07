# smoothcon

JAX-native basis and penalty construction for smooth terms.

```python
import jax.numpy as jnp
import smoothcon

x = jnp.linspace(0.0, 1.0, 100)
smooth = smoothcon.pspline(x, k=20, degree=3, penalty_order=2)
```

See the [API reference](api.md) for the public constructors and `Smooth`
transformations.
