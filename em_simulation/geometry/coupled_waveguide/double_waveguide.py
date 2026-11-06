"""Shared section binding for two-waveguide geometries."""

from ..geometry import Geometry


class DoubleWaveguide(Geometry):
    """Bind an explicit section model to a two-waveguide dataset.

    Models supply describe_cross_section(parameters) and validate_configuration(stored).
    The shared section data consists of the stack and core polygons, with stable
    guide_id and region_id values supplied by the model. Shape-specific fields
    such as gap are outside this contract. Subclasses define the path and axes.
    """

    def __init__(self, dataset, section_model, limit_mode_number=0, verbose=True):
        super().__init__(dataset, limit_mode_number=limit_mode_number, verbose=verbose)
        if not callable(getattr(section_model, "describe_cross_section", None)):
            raise TypeError("section_model must provide describe_cross_section(parameters)")
        self._section_model = section_model

        stored = getattr(dataset, "section_model_configuration", None)
        if stored is None:
            stored = getattr(getattr(dataset, "data_info", None), "section_model_configuration", None)
        if stored is not None:
            section_model.validate_configuration(stored)

    @property
    def section_model(self):
        return self._section_model

    def describe_cross_section(self, parameters):
        """Return the bound model's section for a mapping of parameter values."""
        return self.section_model.describe_cross_section(parameters)
