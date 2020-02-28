class DuplicateFeature(Exception):
    """All feature must be specified exactly once."""
    pass


class UnexpectedFeatureInInteractions(Exception):
    """All features in interaction must be specified."""
    pass


class InteractionWithOnlyOneVariable(Exception):
    """At least two features are need for interaction."""
    pass
