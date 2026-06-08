"""GLM Algorithm Arena — Phase 2 solver registry.

Import all 22 concrete solvers and expose them in ALL_SOLVERS for the arena notebook.
Solvers that have not yet been implemented are silently skipped so that partially
implemented batches can still be imported.
"""
from __future__ import annotations

_missing: list[str] = []

def _try_import(name: str, module: str, cls: str):
    try:
        import importlib
        mod = importlib.import_module(module, package=__name__)
        return getattr(mod, cls)
    except (ImportError, ModuleNotFoundError, AttributeError):
        _missing.append(name)
        return None


from .classical.ols import OLSSolver
from .classical.ridge import RidgeSolver
from .classical.glm_irls import GLMIRLSSolver
from .path.lars import LARSSolver
from .first_order.ista import ISTASolver
from .first_order.fista import FISTASolver

LassoCDSolver = _try_import("LassoCDSolver", ".penalized.lasso_cd", "LassoCDSolver")
ElasticNetSolver = _try_import("ElasticNetSolver", ".penalized.elastic_net", "ElasticNetSolver")
AdaptiveLassoSolver = _try_import("AdaptiveLassoSolver", ".penalized.adaptive_lasso", "AdaptiveLassoSolver")
SCADLLASolver = _try_import("SCADLLASolver", ".penalized.scad_lla", "SCADLLASolver")
MCPCDSolver = _try_import("MCPCDSolver", ".penalized.mcp_cd", "MCPCDSolver")
GroupLassoSolver = _try_import("GroupLassoSolver", ".penalized.group_lasso", "GroupLassoSolver")
FusedLassoSolver = _try_import("FusedLassoSolver", ".penalized.fused_lasso", "FusedLassoSolver")
DebiasedLassoSolver = _try_import("DebiasedLassoSolver", ".inference.debiased_lasso", "DebiasedLassoSolver")
DecorrelatedScoreSolver = _try_import("DecorrelatedScoreSolver", ".inference.decorrelated_score", "DecorrelatedScoreSolver")
SGDSolver = _try_import("SGDSolver", ".online.sgd", "SGDSolver")
ImplicitSGDSolver = _try_import("ImplicitSGDSolver", ".online.implicit_sgd", "ImplicitSGDSolver")
AdaGradSolver = _try_import("AdaGradSolver", ".online.adagrad", "AdaGradSolver")
FOBOSSolver = _try_import("FOBOSSolver", ".online.fobos", "FOBOSSolver")
RDASolver = _try_import("RDASolver", ".online.rda", "RDASolver")
TruncatedGradientSolver = _try_import("TruncatedGradientSolver", ".online.truncated_gradient", "TruncatedGradientSolver")
RenewableGLMSolver = _try_import("RenewableGLMSolver", ".online.renewable_glm", "RenewableGLMSolver")

ALL_SOLVERS = [
    s for s in [
        OLSSolver, RidgeSolver, GLMIRLSSolver,
        LassoCDSolver, ElasticNetSolver, AdaptiveLassoSolver,
        SCADLLASolver, MCPCDSolver, GroupLassoSolver, FusedLassoSolver,
        LARSSolver,
        ISTASolver, FISTASolver,
        DebiasedLassoSolver, DecorrelatedScoreSolver,
        SGDSolver, ImplicitSGDSolver, AdaGradSolver,
        FOBOSSolver, RDASolver, TruncatedGradientSolver, RenewableGLMSolver,
    ] if s is not None
]
