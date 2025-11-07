from importlib import import_module
import inspect

# Map user-facing names to "module_path:ClassName"
_opt_map = {
    "DAMSCo": "optimizers.DAMSCo:DAMSCo",
    "DaSHCo": "optimizers.DaSHCo:DaSHCo",
    "ESMuon": "optimizers.Muon_exactSVD:MuonexactSVD"
}

def _load_class(spec: str):
    module_path, _, class_name = spec.partition(":")
    mod = import_module(module_path)
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise ImportError(f"Class '{class_name}' not found in '{module_path}'") from e
    
def _safe_instantiate(cls, *args, **kwargs):
    # Avoid "multiple values for argument 'X'" by dropping kw that are filled by positional args
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters.values()
              if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    positional_names = [p.name for p in params[:len(args)]]
    for name in positional_names:
        kwargs.pop(name, None)
    return cls(*args, **kwargs)

def get_optimizer(name, *args, **kwargs):
    try:
        spec = _opt_map[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_opt_map)}")
    cls = _load_class(spec)

    return _safe_instantiate(cls, *args, **kwargs)

