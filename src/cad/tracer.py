"""Runtime instrumentation and method interception for CadQuery execution.

Intercepts solid-changing operations dynamically at runtime to extract true
intermediate states S_k and actions A_k preserving fluent CadQuery chaining.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cadquery as cq

from src.cad.action_extractor import capture_action_context
from src.cad.brep_extractor import extract_brep_state
from src.cad.renderer import HeadlessCadRenderer
from src.data.schema import create_empty_state
from src.data.vocabulary import SOLID_COMMIT_OPS, is_solid_commit_op

logger = logging.getLogger(__name__)


class CadQueryRuntimeTracer:
    """Context manager intercepting solid-modifying CadQuery Workplane methods."""

    def __init__(
        self,
        render_images: bool = True,
        renderer: Optional[HeadlessCadRenderer] = None,
        retain_solids: bool = False,
        reference_bbox: Optional[Union[List[float], Tuple[float, ...], Any]] = None,
        uv_grid_size: int = 16,
        curve_samples: int = 16,
    ) -> None:
        self.render_images = render_images
        self.renderer = renderer or HeadlessCadRenderer()
        self.retain_solids = retain_solids
        self.reference_bbox = reference_bbox
        self.uv_grid_size = uv_grid_size
        self.curve_samples = curve_samples

        self.states: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.solids: List[Any] = []
        self.current_solid: Optional[Any] = None
        self.unsupported_ops: List[str] = []

        self._orig_methods: Dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> "CadQueryRuntimeTracer":
        # Always initialize S_0 as the explicit empty workspace
        s_0 = create_empty_state(
            uv_grid_size=self.uv_grid_size,
            curve_samples=self.curve_samples,
            image_size=self.renderer.image_size if self.renderer else 224,
        )
        if self.render_images:
            s_0["images"] = self.renderer.render_empty()

        self.states = [s_0]
        self.actions = []
        self.solids = []
        self.current_solid = None
        self.unsupported_ops = []

        # Monkey-patch solid commit operations on cq.Workplane
        for op in SOLID_COMMIT_OPS:
            if hasattr(cq.Workplane, op):
                orig = getattr(cq.Workplane, op)
                self._orig_methods[op] = orig
                setattr(cq.Workplane, op, self._make_wrapper(op, orig))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Always restore original methods cleanly
        for op, orig in self._orig_methods.items():
            try:
                setattr(cq.Workplane, op, orig)
            except Exception:
                pass
        self._orig_methods.clear()

    def _make_wrapper(self, op_name: str, orig_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper capturing action context before op and state after op."""

        def wrapper(wp_self: Any, *args: Any, **kwargs: Any) -> Any:
            # 1. Pre-operation hook: capture action A_k BEFORE topology changes
            try:
                action_context = capture_action_context(
                    workplane=wp_self,
                    operation_name=op_name,
                    args=args,
                    kwargs=kwargs,
                    prev_state_shape=self.current_solid,
                )
            except Exception as e:
                logger.debug(f"Action context capture error on {op_name}: {e}")
                action_context = {
                    "cmd_id": 0,
                    "params_raw": None,
                    "params": None,
                    "param_mask": None,
                    "ref_kind": 0,
                    "ref_entity_indices": None,
                    "ref_points": None,
                    "ref_directions": None,
                    "operation_name": op_name,
                }

            # 2. Execute original CadQuery operation
            result = orig_func(wp_self, *args, **kwargs)

            # 3. Post-operation hook: extract resulting solid state S_{k+1}
            try:
                if self.render_images:
                    imgs_tensor = self.renderer.render(result, reference_bbox=self.reference_bbox)
                else:
                    imgs_tensor = self.renderer.render_empty()

                next_state = extract_brep_state(
                    result,
                    uv_grid_size=self.uv_grid_size,
                    curve_samples=self.curve_samples,
                    images_tensor=imgs_tensor,
                )

                # Append transition
                self.actions.append(action_context)
                self.states.append(next_state)
                self.current_solid = result

                if self.retain_solids:
                    try:
                        self.solids.append(result.val())
                    except Exception:
                        self.solids.append(result)

            except Exception as e:
                logger.warning(f"Error extracting B-Rep state after {op_name}: {e}")
                raise

            return result

        return wrapper

    def re_render_trajectory_with_bbox(
        self,
        reference_bbox: Optional[Union[List[float], Tuple[float, ...], Any]] = None,
    ) -> None:
        """Re-render all intermediate states with a unified camera bounding box.

        This ensures consistent perspective scaling across the entire construction sequence,
        avoiding artificial zoom-in/zoom-out across progressive extrusion steps.
        Requires retain_solids=True during tracing.
        """
        if not self.render_images or not self.solids:
            return

        if reference_bbox is None:
            # Anchor to the bounding box of the final generated solid S_K
            last_state = self.states[-1]
            if "bounding_box" in last_state and last_state["bounding_box"] is not None:
                reference_bbox = last_state["bounding_box"]

        if reference_bbox is None:
            return

        self.reference_bbox = reference_bbox
        for idx, solid in enumerate(self.solids):
            state_idx = idx + 1
            if state_idx < len(self.states):
                self.states[state_idx]["images"] = self.renderer.render(
                    solid, reference_bbox=reference_bbox
                )

    def get_trajectory_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return the extracted (states, actions) sequence."""
        return self.states, self.actions

    def get_solids(self) -> List[Any]:
        """Return the list of in-memory CadQuery/OpenCASCADE solids at each step."""
        return self.solids
