from signal_model.spline_field.bspline import BSplineRainField
from signal_model.spline_field.bspline_sampler import BSplineSampler, BSplineMixerSampler
from signal_model.base_field_sampler import SamplerConfig


def generate_sampler(in_filed: BSplineRainField, in_sensors, in_sampler_config: SamplerConfig):
    if isinstance(in_filed, BSplineRainField):
        return BSplineSampler(in_filed, in_sensors, in_sampler_config)
    else:
        raise NotImplemented


def generate_mixed_sampler(in_field,in_point, in_link, in_sampler_point: SamplerConfig, in_sampler_link: SamplerConfig):
    return BSplineMixerSampler(in_field, in_point, in_link, in_sampler_point, in_sampler_link)