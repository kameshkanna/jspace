---
name: New model support
about: HookedModel fails to discover layers for a model you want to use
labels: model-support
---

**Model ID**
(e.g. `mistralai/Mistral-7B-Instruct-v0.3`)

**Error**
```
# paste the RuntimeError from HookedModel._discover_layers
```

**Layer path**
If you know the path to the transformer blocks in this model's architecture, add it here.
You can find it by running:
```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("your/model", torch_dtype="auto")
print([name for name, _ in m.named_modules() if "layer" in name.lower()][:10])
```

**Did you try adding it to `_discover_layers`?**
If so, paste your diff and confirm workspace detection produced sensible output.
