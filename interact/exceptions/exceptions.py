class NoFeatureProvided(Exception):
    """At least one feature must be provided."""
    pass


class NoFeatureProvidedInInteraction(Exception):
    """At least one feature must be provided."""
    pass


class DuplicateFeature(Exception):
    """All feature must be specified exactly once."""
    pass


class UnexpectedFeatureInInteractions(Exception):
    """All features in interaction must be specified."""
    pass


class SingleDenseFeatureInInteraction(Exception):
    """At least two features are need for interaction."""
    pass
